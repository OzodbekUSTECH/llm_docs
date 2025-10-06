

import asyncio
import time
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from ollama import AsyncClient

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse, ToolCall, Source
from app.services.chat_storage import chat_storage
from app.interactors.chat.system_prompts import STRICT_RAG_PROMPT

# Хранилище истории для каждого чата
chat_histories: Dict[str, List[Dict[str, Any]]] = {}


def clear_chat_history(chat_id: str) -> None:
    """Очищает историю чата (оставляет только system prompt)"""
    if chat_id in chat_histories:
        chat_histories[chat_id] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        print(f"[CHAT] History cleared for chat {chat_id}")


def reset_all_chat_histories() -> None:
    """Полностью очищает все истории чатов"""
    chat_histories.clear()
    print("[CHAT] All chat histories cleared")

# System prompt - используем строгий режим для минимизации галлюцинаций
# Можно заменить на другие из system_prompts.py:
# - BALANCED_PROMPT - сбалансированный (документы + общие знания)
# - FRIENDLY_PROMPT - дружелюбный режим
# - TECHNICAL_EXPERT_PROMPT - для технической документации
# - RUSSIAN_STRICT_PROMPT - строгий режим на русском языке
SYSTEM_PROMPT = STRICT_RAG_PROMPT

class GenerateAnswerInteractor:
    """Интерактор для генерации ответов с использованием инструментов LLM"""
    
    def __init__(
        self, 
        llm: AsyncClient
    ):
        self.llm = llm
        
    async def execute(
        self, 
        request: GenerateAnswerRequest,
        chat_id: Optional[str] = None
    ) -> GeneratedAnswerResponse:
        from app.utils.tools.registry import available_tools_dict, available_tools
        
        start_time = time.time()
        message_id = str(uuid.uuid4())
        model_used = "gpt-oss:20b"
        
        # Initialize response data
        tool_calls = []
        sources = []
        reasoning = None
        final_content = ""
        
        try:
            # Получаем или создаем историю для чата
            if chat_id not in chat_histories:
                chat_histories[chat_id] = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]
            chat_history = chat_histories[chat_id]
            
            # Сохраняем сообщение пользователя
            chat_storage.add_message(chat_id, "user", request.message)
            
            # Add user message to history
            chat_history.append({
                "role": "user",
                "content": request.message,
            })
            
            print(f"[CHAT] Processing message for chat {chat_id}: {request.message}")
            
            # Если это первое сообщение, будем генерировать название позже
            # System prompt (1) + user message (1) = 2 messages for first user interaction
            is_first_message = len(chat_history) == 2
            
            # Agentic loop: allow multiple rounds of tool calling
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"[CHAT] Agent iteration {iteration}/{max_iterations}")
                
                # LLM call with tools (with timeout)
                try:
                    response = await asyncio.wait_for(
                        self.llm.chat(
                            model=model_used,
                            messages=chat_history,
                            options={
                                "temperature": 0,  # Make responses more deterministic
                                "num_ctx": 4096  # Limit context window to prevent memory issues
                            },
                            tools=available_tools
                        ),
                        timeout=60.0  # 60 second timeout
                    )
                except asyncio.TimeoutError:
                    print(f"[CHAT] LLM call timed out after 60 seconds")
                    final_content = "Извините, обработка запроса заняла слишком много времени. Попробуйте упростить вопрос."
                    break
                
                chat_history.append(response.message)
                
                # If no tool calls, LLM has generated final response - break
                if not response.message.tool_calls:
                    print("[CHAT] No tool calls - final response generated")
                    break
                
                # Process tool calls
                print(f"[CHAT] Found {len(response.message.tool_calls)} tool calls")
                
                has_successful_tool = False
                for tool in response.message.tool_calls:
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        success=False
                    )
                    
                    try:
                        if function_to_call := available_tools_dict.get(tool.function.name):
                            print(f'[TOOL] Calling: {tool.function.name}')
                            print(f'[TOOL] Arguments: {tool.function.arguments}')
                            
                            func_output = await function_to_call(**tool.function.arguments)
                            tool_call.output = str(func_output)
                            tool_call.success = True
                            has_successful_tool = True
                            
                            print(f'[TOOL] Output: {func_output}')
                            
                            # Extract sources if it's a search function
                            if tool.function.name == "search_documents" and isinstance(func_output, list):
                                for result in func_output:
                                    if isinstance(result, dict) and 'best_chunks' in result:
                                        for chunk in result['best_chunks']:
                                            sources.append(Source(
                                                filename=result.get('filename', 'Unknown'),
                                                content=chunk.get('content', ''),
                                                similarity=chunk.get('similarity', 0.0),
                                                chunk_index=chunk.get('chunk_index', 0)
                                            ))
                            
                            # Add tool result to chat history with explicit instruction
                            tool_response = str(func_output)
                            
                            # Ограничиваем размер tool response для предотвращения перегрузки
                            max_tool_response_length = 8000
                            if len(tool_response) > max_tool_response_length:
                                tool_response = tool_response[:max_tool_response_length] + "\n\n[... response truncated due to length ...]"
                                print(f"[TOOL] Response truncated from {len(str(func_output))} to {max_tool_response_length} chars")
                            
                            chat_history.append({
                                'role': 'tool', 
                                'content': f"TOOL OUTPUT - USE ONLY THIS INFORMATION:\n\n{tool_response}\n\nIMPORTANT: Base your answer ONLY on the information above. Do NOT add information from your training data.", 
                                'tool_name': tool.function.name
                            })
                        else:
                            tool_call.error = f"Function {tool.function.name} not found"
                            print(f'[TOOL] Function {tool.function.name} not found')
                            
                    except Exception as e:
                        tool_call.error = str(e)
                        tool_call.success = False
                        print(f'[TOOL] Error calling {tool.function.name}: {e}')
                    
                    tool_calls.append(tool_call)
                
                # If no successful tools, break to avoid infinite loop
                if not has_successful_tool:
                    print("[CHAT] No successful tool calls - breaking loop")
                    break
                
                # Update reasoning
                successful_tools = [tc.name for tc in tool_calls if tc.success]
                reasoning = f"Used {len(successful_tools)} tool(s) to gather information: {', '.join(successful_tools)}"
                
                # Continue loop - LLM will decide if more tools needed or provide final answer
            
            # Set final content
            final_content = response.message.content if response.message.content else "Извините, я не смог сгенерировать ответ на основе найденной информации. Попробуйте переформулировать вопрос."
            
            # Сохраняем ответ ассистента с metadata
            metadata = {
                'sources': [s.model_dump() for s in sources],
                'tool_calls': [tc.model_dump() for tc in tool_calls],
                'reasoning': reasoning
            }
            print(f"[CHAT] Saving message with metadata: sources={len(sources)}, tool_calls={len(tool_calls)}")
            chat_storage.add_message(chat_id, "assistant", final_content, metadata=metadata)
            
            # Генерируем название для первого сообщения
            if is_first_message and chat_id:
                asyncio.create_task(self._generate_chat_title(chat_id, request.message, final_content))
            
        except Exception as e:
            print(f"[CHAT] Error during processing: {e}")
            final_content = f"Произошла ошибка при обработке запроса: {str(e)}"
            reasoning = f"Error occurred: {str(e)}"
            # Сохраняем даже ошибочный ответ
            if chat_id:
                chat_storage.add_message(chat_id, "assistant", final_content)
        
        processing_time = time.time() - start_time
        
        # Create structured response
        response_data = GeneratedAnswerResponse(
            message_id=message_id,
            role="assistant",
            content=final_content,
            sources=sources,
            tool_calls=tool_calls,
            reasoning=reasoning,
            processing_time=round(processing_time, 2),
            model_used=model_used,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[CHAT] Generated structured response: {response_data}")
        return response_data
    
    async def _generate_chat_title(self, chat_id: str, user_message: str, assistant_response: str):
        """
        Генерирует короткое название чата на основе первого сообщения (как в ChatGPT)
        """
        try:
            print(f"[CHAT] Generating title for chat {chat_id}")
            
            prompt = f"""Based on this conversation, generate a short, descriptive title (max 6 words) in Russian.
            
User: {user_message}
Assistant: {assistant_response[:200]}

Title should be concise and capture the main topic. Respond with ONLY the title, nothing else."""
            
            response = await self.llm.chat(
                model="gpt-oss:20b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7}
            )
            
            title = response.message.content.strip().strip('"').strip("'")
            
            # Ограничиваем длину
            if len(title) > 60:
                title = title[:60] + "..."
            
            # Обновляем название чата
            chat_storage.update_chat_title(chat_id, title)
            print(f"[CHAT] Generated title: {title}")
            
        except Exception as e:
            print(f"[CHAT] Error generating title: {e}")
    
    async def execute_stream(
        self, 
        request: GenerateAnswerRequest,
        chat_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming версия генерации ответов с использованием инструментов LLM
        Отправляет события в формате Server-Sent Events (SSE)
        """
        from app.utils.tools.registry import available_tools_dict, available_tools
        
        start_time = time.time()
        message_id = str(uuid.uuid4())
        model_used = "gpt-oss:20b"
        
        # Initialize response data
        tool_calls = []
        sources = []
        reasoning = None
        final_content = ""
        
        def send_event(event_type: str, data: Dict[str, Any]) -> str:
            """Форматирует событие для SSE"""
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"
        
        try:
            # Получаем или создаем историю для чата
            if chat_id not in chat_histories:
                chat_histories[chat_id] = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]
            chat_history = chat_histories[chat_id]
            
            # Сохраняем сообщение пользователя
            chat_storage.add_message(chat_id, "user", request.message)
            
            # Add user message to history
            chat_history.append({
                "role": "user",
                "content": request.message,
            })
            
            print(f"[CHAT] Processing message for chat {chat_id}: {request.message}")
            
            # Если это первое сообщение, будем генерировать название позже
            # System prompt (1) + user message (1) = 2 messages for first user interaction
            is_first_message = len(chat_history) == 2
            
            # Send start event
            yield send_event('start', {
                'message_id': message_id,
                'chat_id': chat_id
            })
            
            # Agentic loop: allow multiple rounds of tool calling
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"[CHAT] Agent iteration {iteration}/{max_iterations}")
                
                yield send_event('iteration', {
                    'iteration': iteration,
                    'max_iterations': max_iterations
                })
                
                # Stream LLM response with tools
                # Ollama returns tool_calls in the LAST chunk when streaming
                print("[CHAT] Streaming LLM response...")
                yield send_event('thinking', {
                    'message': 'Думаю...'
                })
                
                accumulated_content = ""
                response = None
                content_started = False
                
                # Streaming LLM call with timeout protection
                try:
                    stream = await asyncio.wait_for(
                        self.llm.chat(
                            model=model_used,
                            messages=chat_history,
                            options={
                                "temperature": 0,
                                "num_ctx": 4096  # Limit context window
                            },
                            tools=available_tools,
                            stream=True
                        ),
                        timeout=5.0  # Short timeout just for initiating stream
                    )
                except asyncio.TimeoutError:
                    print(f"[CHAT] Stream initialization timed out")
                    final_content = "Извините, не удалось начать генерацию ответа. Попробуйте ещё раз."
                    break
                
                chunk_count = 0
                max_chunks = 500  # Limit to prevent infinite streams
                
                try:
                    async for chunk in stream:
                        chunk_count += 1
                        if chunk_count > max_chunks:
                            print(f"[CHAT] Max chunks limit reached ({max_chunks}), breaking stream")
                            break
                        
                        # Accumulate response
                        response = chunk
                        
                        # Stream content if available
                        if chunk.message.content:
                            if not content_started:
                                content_started = True
                                yield send_event('content_start', {
                                    'message': 'Генерирую ответ...'
                                })
                            
                            accumulated_content += chunk.message.content
                            
                            # Limit accumulated content length
                            if len(accumulated_content) > 10000:
                                print(f"[CHAT] Accumulated content too long, stopping stream")
                                break
                            
                            yield send_event('content_chunk', {
                                'chunk': chunk.message.content
                            })
                except Exception as stream_error:
                    print(f"[CHAT] Error during streaming: {stream_error}")
                    # Continue processing with what we have
                
                # Check last chunk for tool calls
                if response and response.message.tool_calls:
                    print(f"[CHAT] Tool calls detected in stream")
                    # Has tool calls - add to history and continue
                    chat_history.append(response.message)
                elif accumulated_content:
                    # No tool calls, content was streamed - this is final answer
                    print(f"[CHAT] Final content streamed: {len(accumulated_content)} chars")
                    chat_history.append({
                        'role': 'assistant',
                        'content': accumulated_content
                    })
                    # Store final content explicitly
                    final_content = accumulated_content
                    break
                else:
                    # No content and no tool calls - something went wrong
                    print("[CHAT] No content or tool calls received")
                    final_content = "Извините, не удалось получить ответ."
                    break
                
                # If we have tool calls, process them
                if not (response and response.message.tool_calls):
                    continue
                
                # Process tool calls
                print(f"[CHAT] Found {len(response.message.tool_calls)} tool calls")
                
                has_successful_tool = False
                for tool in response.message.tool_calls:
                    tool_call = ToolCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                        success=False
                    )
                    
                    # Send tool call start event
                    yield send_event('tool_call_start', {
                        'tool_name': tool.function.name,
                        'arguments': tool.function.arguments
                    })
                    
                    try:
                        if function_to_call := available_tools_dict.get(tool.function.name):
                            print(f'[TOOL] Calling: {tool.function.name}')
                            print(f'[TOOL] Arguments: {tool.function.arguments}')
                            
                            func_output = await function_to_call(**tool.function.arguments)
                            tool_call.output = str(func_output)
                            tool_call.success = True
                            has_successful_tool = True
                            
                            print(f'[TOOL] Output: {func_output}')
                            
                            # Send tool call success event
                            yield send_event('tool_call_success', {
                                'tool_name': tool.function.name,
                                'output': str(func_output)[:500] + '...' if len(str(func_output)) > 500 else str(func_output)
                            })
                            
                            # Extract sources if it's a search function
                            if tool.function.name == "search_documents" and isinstance(func_output, list):
                                for result in func_output:
                                    if isinstance(result, dict) and 'best_chunks' in result:
                                        for chunk in result['best_chunks']:
                                            sources.append(Source(
                                                filename=result.get('filename', 'Unknown'),
                                                content=chunk.get('content', ''),
                                                similarity=chunk.get('similarity', 0.0),
                                                chunk_index=chunk.get('chunk_index', 0)
                                            ))
                            
                            # Add tool result to chat history with explicit instruction
                            tool_response = str(func_output)
                            
                            # Ограничиваем размер tool response для предотвращения перегрузки
                            max_tool_response_length = 8000
                            if len(tool_response) > max_tool_response_length:
                                tool_response = tool_response[:max_tool_response_length] + "\n\n[... response truncated due to length ...]"
                                print(f"[TOOL] Response truncated from {len(str(func_output))} to {max_tool_response_length} chars")
                            
                            chat_history.append({
                                'role': 'tool', 
                                'content': f"TOOL OUTPUT - USE ONLY THIS INFORMATION:\n\n{tool_response}\n\nIMPORTANT: Base your answer ONLY on the information above. Do NOT add information from your training data.", 
                                'tool_name': tool.function.name
                            })
                        else:
                            tool_call.error = f"Function {tool.function.name} not found"
                            print(f'[TOOL] Function {tool.function.name} not found')
                            yield send_event('tool_call_error', {
                                'tool_name': tool.function.name,
                                'error': tool_call.error
                            })
                            
                    except Exception as e:
                        tool_call.error = str(e)
                        tool_call.success = False
                        print(f'[TOOL] Error calling {tool.function.name}: {e}')
                        yield send_event('tool_call_error', {
                            'tool_name': tool.function.name,
                            'error': str(e)
                        })
                    
                    tool_calls.append(tool_call)
                
                # If no successful tools, break to avoid infinite loop
                if not has_successful_tool:
                    print("[CHAT] No successful tool calls - breaking loop")
                    break
                
                # Update reasoning
                successful_tools = [tc.name for tc in tool_calls if tc.success]
                reasoning = f"Used {len(successful_tools)} tool(s) to gather information: {', '.join(successful_tools)}"
                
                # Continue loop - LLM will decide if more tools needed or provide final answer
            
            # Set final content if not already set during streaming
            if not final_content:
                if response and hasattr(response, 'message') and hasattr(response.message, 'content'):
                    final_content = response.message.content or "Извините, я не смог сгенерировать ответ на основе найденной информации. Попробуйте переформулировать вопрос."
                else:
                    final_content = "Извините, я не смог сгенерировать ответ на основе найденной информации. Попробуйте переформулировать вопрос."
            
            print(f"[CHAT] Final content to save: {len(final_content)} chars")
            
            # Сохраняем ответ ассистента с metadata
            metadata = {
                'sources': [s.model_dump() for s in sources],
                'tool_calls': [tc.model_dump() for tc in tool_calls],
                'reasoning': reasoning
            }
            print(f"[CHAT] Saving message with metadata: sources={len(sources)}, tool_calls={len(tool_calls)}")
            chat_storage.add_message(chat_id, "assistant", final_content, metadata=metadata)
            
            # Send final response
            processing_time = time.time() - start_time
            
            response_data = GeneratedAnswerResponse(
                message_id=message_id,
                role="assistant",
                content=final_content,
                sources=sources,
                tool_calls=tool_calls,
                reasoning=reasoning,
                processing_time=round(processing_time, 2),
                model_used=model_used,
                timestamp=datetime.now().isoformat()
            )
            
            complete_event = {
                'message_id': message_id,
                'role': 'assistant',
                'content': final_content,
                'sources': [s.model_dump() for s in sources],
                'tool_calls': [tc.model_dump() for tc in tool_calls],
                'reasoning': reasoning,
                'processing_time': round(processing_time, 2),
                'model_used': model_used,
                'timestamp': datetime.now().isoformat()
            }
            print(f"[CHAT] Sending complete event with {len(final_content)} chars, {len(sources)} sources, {len(tool_calls)} tool_calls")
            yield send_event('complete', complete_event)
            
            # Генерируем название для первого сообщения
            if is_first_message and chat_id:
                asyncio.create_task(self._generate_chat_title(chat_id, request.message, final_content))
            
        except Exception as e:
            print(f"[CHAT] Error during processing: {e}")
            final_content = f"Произошла ошибка при обработке запроса: {str(e)}"
            
            # Сохраняем даже ошибочный ответ
            if chat_id:
                chat_storage.add_message(chat_id, "assistant", final_content)
            
            yield send_event('error', {
                'message': final_content,
                'error': str(e)
            })