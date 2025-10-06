import asyncio
import time
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse, ToolCall, Source
from app.services.chat_storage import chat_storage

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


# System prompt - strict RAG mode for document-based responses
SYSTEM_PROMPT = """You are an AI assistant with access to a document knowledge base through specialized tools.

CRITICAL RULES - FOLLOW STRICTLY:
1. **NEVER fabricate or invent information** - use ONLY facts from tool outputs
2. **ALWAYS use search_documents** when asked about specific information, facts, or topics that might be in documents
3. **ANSWER ONLY based on retrieved information** - if tools return no relevant data, say "I don't have information about this in the available documents"
4. **QUOTE from tool outputs** - when you receive tool results, read them carefully and use ONLY that information
5. **Don't assume or speculate** - if information is incomplete or unclear in retrieved data, explicitly state this
6. **Use tools proactively** - when the user asks a question requiring information lookup, immediately use search_documents
7. **IGNORE your training data** - trust ONLY tool outputs, not general knowledge for factual questions

AVAILABLE TOOLS:
- search_documents: Search through uploaded documents for relevant information. Use this for ANY factual question.
- get_document_by_id: Get full document content if you need more details after searching.

STRICT RESPONSE PROTOCOL:
1. When you receive tool output, READ IT CAREFULLY
2. Look for the answer in "relevant_content" and "best_chunks" fields
3. If the answer is there, extract it and respond with ONLY that information
4. If the answer is NOT there, say "I couldn't find information about [topic] in the available documents"
5. NEVER provide information that is not explicitly present in the tool output

FORBIDDEN ACTIONS:
- ❌ NEVER invent URLs, links, or file paths
- ❌ NEVER mention documents that weren't in the tool output
- ❌ NEVER use general knowledge for factual questions
- ❌ NEVER provide answers if the tool output doesn't contain the information

Remember: It's better to say "I don't know" than to provide incorrect information. If tool output doesn't answer the question, admit it clearly.

Be professional, accurate, and respond in the same language as the user's question."""

# OpenAI API tools definition
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search for relevant documents using semantic vector search. Returns document IDs and relevant content chunks with high similarity scores. This tool performs intelligent semantic search across all uploaded documents and returns the most relevant information. Use this tool to find any factual information, data, or content from documents (e.g., company names, owners, directors, contracts, specifications, reports, etc.). The tool returns full relevant chunks without truncation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query. Be specific and descriptive. Examples: 'owner of Floriana Impex company', 'vessel name and specifications', 'contract details and terms', 'company registration information', 'financial data for Q4 2023'. The more specific your query, the better the results."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return. Higher values provide more context but may include less relevant results. Recommended: 3-5 for focused searches, 5-10 for comprehensive searches. Default is 10 for maximum information coverage.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_by_id",
            "description": "Retrieve the complete full text content of a specific document by its ID. Use this tool after search_documents when you need the entire document content, not just relevant excerpts. This returns the full untruncated document text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document UUID obtained from search_documents results. Each search result includes an 'id' field with the document UUID."
                    }
                },
                "required": ["document_id"]
            }
        }
    }
]


class OpenAIGenerateInteractor:
    """Интерактор для генерации ответов с использованием OpenAI API с поддержкой инструментов для работы с документами"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        
    async def execute(
        self, 
        request: GenerateAnswerRequest,
        chat_id: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ) -> GeneratedAnswerResponse:
        """
        Генерация ответа с использованием OpenAI API (без streaming)
        
        Args:
            request: Запрос с сообщением пользователя
            chat_id: Идентификатор чата
            model: Модель OpenAI
        """
        start_time = time.time()
        message_id = str(uuid.uuid4())
        
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
            
            # Если это первое сообщение
            is_first_message = len(chat_history) == 2
            
            # OpenAI API call
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=model,
                    messages=chat_history,
                    temperature=0.7,
                    max_tokens=4000  # Увеличено для более полных ответов
                ),
                timeout=60.0
            )
            
            response_message = response.choices[0].message
            final_content = response_message.content or ""
            
            # Add assistant message to history
            chat_history.append({
                "role": "assistant",
                "content": final_content
            })
            
            # Сохраняем ответ ассистента
            chat_storage.add_message(chat_id, "assistant", final_content)
            
            # Генерируем название для первого сообщения
            if is_first_message and chat_id:
                asyncio.create_task(self._generate_chat_title(chat_id, request.message, final_content, model))
            
        except asyncio.TimeoutError:
            print(f"[CHAT] OpenAI API call timed out")
            final_content = "Извините, обработка запроса заняла слишком много времени. Попробуйте упростить вопрос."
            chat_storage.add_message(chat_id, "assistant", final_content)
        except Exception as e:
            print(f"[CHAT] Error during processing: {e}")
            final_content = f"Произошла ошибка при обработке запроса: {str(e)}"
            chat_storage.add_message(chat_id, "assistant", final_content)
        
        processing_time = time.time() - start_time
        
        # Create structured response
        response_data = GeneratedAnswerResponse(
            message_id=message_id,
            role="assistant",
            content=final_content,
            sources=[],
            tool_calls=[],
            reasoning=None,
            processing_time=round(processing_time, 2),
            model_used=model,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"[CHAT] Generated response in {processing_time:.2f}s")
        return response_data
    
    async def _generate_chat_title(self, chat_id: str, user_message: str, assistant_response: str, model: str):
        """
        Генерирует короткое название чата на основе первого сообщения
        """
        try:
            print(f"[CHAT] Generating title for chat {chat_id}")
            
            prompt = f"""На основе этого разговора создай короткое название (максимум 6 слов) на русском языке.

Пользователь: {user_message}
Ассистент: {assistant_response[:200]}

Название должно быть кратким и отражать главную тему. Отвечай ТОЛЬКО названием, ничего больше."""
            
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=20
            )
            
            title = response.choices[0].message.content.strip().strip('"').strip("'")
            
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
        chat_id: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ) -> AsyncGenerator[str, None]:
        """
        Streaming версия генерации ответов с использованием OpenAI API с тулами
        Отправляет события в формате Server-Sent Events (SSE)
        
        Args:
            request: Запрос с сообщением пользователя
            chat_id: Идентификатор чата
            model: Модель OpenAI
        """
        from app.utils.tools.registry import available_tools_dict
        
        start_time = time.time()
        message_id = str(uuid.uuid4())
        accumulated_content = ""
        tool_calls_list = []
        sources = []
        
        def send_event(event_type: str, data: Dict[str, Any]) -> str:
            """Форматирует событие для SSE"""
            return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"
        
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
            
            print(f"[CHAT] Processing streaming message for chat {chat_id}: {request.message}")
            
            # Если это первое сообщение
            is_first_message = len(chat_history) == 2
            
            # Send start event
            yield send_event('start', {
                'message_id': message_id,
                'chat_id': chat_id
            })
            
            # Agentic loop для работы с инструментами
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"[CHAT] Agent iteration {iteration}/{max_iterations}")
                
                yield send_event('iteration', {
                    'iteration': iteration,
                    'max_iterations': max_iterations
                })
                
                yield send_event('thinking', {
                    'message': 'Анализирую запрос...'
                })
                
                # Stream OpenAI response with tools
                try:
                    stream = await asyncio.wait_for(
                        self.openai_client.chat.completions.create(
                            model=model,
                            messages=chat_history,
                            temperature=0,
                            max_tokens=4000,  # Увеличено для более полных ответов
                            tools=OPENAI_TOOLS,
                            tool_choice="auto",
                            stream=True
                        ),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    print(f"[CHAT] Stream initialization timed out")
                    yield send_event('error', {
                        'message': 'Не удалось начать генерацию ответа. Попробуйте ещё раз.',
                        'error': 'Timeout'
                    })
                    return
                
                content_started = False
                accumulated_tool_calls = []
                
                try:
                    async for chunk in stream:
                        delta = chunk.choices[0].delta
                        
                        # Stream content if available
                        if delta.content:
                            if not content_started:
                                content_started = True
                                yield send_event('content_start', {
                                    'message': 'Генерирую ответ...'
                                })
                            
                            accumulated_content += delta.content
                            
                            yield send_event('content_chunk', {
                                'chunk': delta.content
                            })
                        
                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tc_chunk in delta.tool_calls:
                                # Extend accumulated_tool_calls if needed
                                while len(accumulated_tool_calls) <= tc_chunk.index:
                                    accumulated_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": ""
                                        }
                                    })
                                
                                tc = accumulated_tool_calls[tc_chunk.index]
                                
                                if tc_chunk.id:
                                    tc["id"] = tc_chunk.id
                                if tc_chunk.function:
                                    if tc_chunk.function.name:
                                        tc["function"]["name"] = tc_chunk.function.name
                                    if tc_chunk.function.arguments:
                                        tc["function"]["arguments"] += tc_chunk.function.arguments
                        
                        # Check if generation is complete
                        if chunk.choices[0].finish_reason:
                            break
                
                except Exception as stream_error:
                    print(f"[CHAT] Error during streaming: {stream_error}")
                    yield send_event('error', {
                        'message': f'Ошибка при генерации: {str(stream_error)}',
                        'error': str(stream_error)
                    })
                    return
                
                # Если есть tool calls - обрабатываем их
                if accumulated_tool_calls:
                    print(f"[CHAT] Tool calls detected: {len(accumulated_tool_calls)}")
                    
                    # Add assistant message with tool calls to history
                    chat_history.append({
                        "role": "assistant",
                        "content": accumulated_content or None,
                        "tool_calls": accumulated_tool_calls
                    })
                    
                    # Process each tool call
                    has_successful_tool = False
                    for tool_call_data in accumulated_tool_calls:
                        tool_call_record = ToolCall(
                            name=tool_call_data["function"]["name"],
                            arguments=json.loads(tool_call_data["function"]["arguments"]),
                            success=False
                        )
                        
                        # Send tool call start event
                        yield send_event('tool_call_start', {
                            'tool_name': tool_call_data["function"]["name"],
                            'arguments': json.loads(tool_call_data["function"]["arguments"])
                        })
                        
                        try:
                            function_name = tool_call_data["function"]["name"]
                            if function_to_call := available_tools_dict.get(function_name):
                                print(f'[TOOL] Calling: {function_name}')
                                print(f'[TOOL] Arguments: {tool_call_data["function"]["arguments"]}')
                                
                                arguments = json.loads(tool_call_data["function"]["arguments"])
                                func_output = await function_to_call(**arguments)
                                tool_call_record.output = str(func_output)
                                tool_call_record.success = True
                                has_successful_tool = True
                                
                                print(f'[TOOL] Output: {func_output}')
                                
                                # Send tool call success event
                                yield send_event('tool_call_success', {
                                    'tool_name': function_name,
                                    'output': str(func_output)[:500] + '...' if len(str(func_output)) > 500 else str(func_output)
                                })
                                
                                # Extract sources if it's search_documents
                                if function_name == "search_documents" and isinstance(func_output, list):
                                    for result in func_output:
                                        if isinstance(result, dict) and 'best_chunks' in result:
                                            for chunk in result['best_chunks']:
                                                sources.append(Source(
                                                    filename=result.get('filename', 'Unknown'),
                                                    content=chunk.get('content', ''),
                                                    similarity=chunk.get('similarity', 0.0),
                                                    chunk_index=chunk.get('chunk_index', 0)
                                                ))
                                
                                # Add FULL tool result to history WITHOUT truncation
                                # LLM needs complete information for accurate answers
                                tool_response = str(func_output)
                                
                                chat_history.append({
                                    'role': 'tool',
                                    'tool_call_id': tool_call_data["id"],
                                    'content': f"TOOL OUTPUT - USE ONLY THIS INFORMATION:\n\n{tool_response}\n\nIMPORTANT: Base your answer STRICTLY on the information above. Do NOT add information from your training data.",
                                    'name': function_name
                                })
                            else:
                                tool_call_record.error = f"Функция {function_name} не найдена"
                                yield send_event('tool_call_error', {
                                    'tool_name': function_name,
                                    'error': tool_call_record.error
                                })
                        
                        except Exception as e:
                            tool_call_record.error = str(e)
                            tool_call_record.success = False
                            print(f'[TOOL] Error: {e}')
                            yield send_event('tool_call_error', {
                                'tool_name': tool_call_data["function"]["name"],
                                'error': str(e)
                            })
                        
                        tool_calls_list.append(tool_call_record)
                    
                    if not has_successful_tool:
                        print("[CHAT] No successful tool calls - breaking")
                        break
                    
                    # Reset content for next iteration
                    accumulated_content = ""
                    
                elif accumulated_content:
                    # No tool calls - final answer
                    print(f"[CHAT] Final content: {len(accumulated_content)} chars")
                    chat_history.append({
                        'role': 'assistant',
                        'content': accumulated_content
                    })
                    break
                else:
                    print("[CHAT] No content or tool calls")
                    break
            
            # Set final content
            if not accumulated_content:
                accumulated_content = "Извините, я не смог сгенерировать ответ. Попробуйте переформулировать вопрос."
            
            # Сохраняем ответ с метаданными
            metadata = {
                'sources': [s.model_dump() for s in sources],
                'tool_calls': [tc.model_dump() for tc in tool_calls_list],
                'reasoning': f"Использовано инструментов: {len(tool_calls_list)}" if tool_calls_list else None
            }
            chat_storage.add_message(chat_id, "assistant", accumulated_content, metadata=metadata)
            
            # Send complete event
            processing_time = time.time() - start_time
            
            complete_event = {
                'message_id': message_id,
                'role': 'assistant',
                'content': accumulated_content,
                'sources': [s.model_dump() for s in sources],
                'tool_calls': [tc.model_dump() for tc in tool_calls_list],
                'reasoning': f"Использовано инструментов: {len(tool_calls_list)}" if tool_calls_list else None,
                'processing_time': round(processing_time, 2),
                'model_used': model,
                'timestamp': datetime.now().isoformat()
            }
            yield send_event('complete', complete_event)
            
            print(f"[CHAT] Streaming completed in {processing_time:.2f}s, {len(accumulated_content)} chars, {len(sources)} sources")
            
            # Генерируем название
            if is_first_message and chat_id:
                asyncio.create_task(self._generate_chat_title(chat_id, request.message, accumulated_content, model))
            
        except Exception as e:
            print(f"[CHAT] Error: {e}")
            yield send_event('error', {
                'message': f'Произошла ошибка: {str(e)}',
                'error': str(e)
            })
            
            if chat_id:
                chat_storage.add_message(chat_id, "assistant", f"Ошибка: {str(e)}")