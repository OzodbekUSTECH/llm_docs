import asyncio
import time
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse, ToolCall, Source
from app.services.chat_storage import chat_storage
from app.utils.openai_tools import OPENAI_TOOLS

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


# System prompt - adaptive RAG mode for document-based responses
SYSTEM_PROMPT = """You are an AI assistant with access to a document knowledge base and business rules through specialized tools.

CRITICAL RULES - FOLLOW STRICTLY:
1. **NEVER fabricate or invent information** - use ONLY facts from tool outputs.
2. **ALWAYS use search_documents** when asked about specific information, facts, or topics that might be in documents.
3. **ALWAYS use search_rules** when asked about business processes, workflows, policies, or how to work with documents.
4. **ANSWER based on retrieved information** – but if your chosen tool returns no relevant data, you MUST:
   - Try other available tools that could help answer the user's question, if any, before saying you have no information.
   - If several tools return no relevant information, consider asking a clarifying follow-up question to the user to better understand their intent or to disambiguate what information is needed.
   - If you are certain all relevant tools and approaches have been exhausted and no information exists, clearly say so (e.g., "I don't have information about this in the available documents/rules. Please specify which document or details you are interested in.").
5. **QUOTE from tool outputs** – when you receive tool results, read them carefully and use ONLY that information.
6. **Don't assume or speculate** – if information is incomplete or unclear in retrieved data, explicitly state this and consider prompting the user for clarification.
7. **Use tools proactively and adaptively** – if the user's question might be answered by multiple tools, you may try each in turn until you find a helpful answer; do not stop after the first failed attempt.
8. **IGNORE your training data** – trust ONLY tool outputs, not general knowledge for factual questions.

AVAILABLE TOOLS:
- search_documents: Search through uploaded documents for relevant information. Returns document metadata and extracted keywords instead of full content to avoid overwhelming the LLM. Use this for ANY factual question about document content.
- get_document_by_id: Get document information and extracted keywords by document ID. Use this to get detailed information about a specific document including all its extracted key data points.
- search_documents_by_keywords: Search for documents by specific extracted keywords (vessel names, invoice numbers, contract details, etc.). Use this when users ask for specific data points like "Find documents with vessel ABC", "Show invoices from company XYZ".
- search_rules: Search through business rules and policies. Use this when users ask about:
  * How to work with documents
  * Business processes and workflows
  * Company policies and procedures
  * Document comparison and analysis methods
  * Compliance requirements
  * Security protocols
  * Data handling procedures
  * Approval processes
  * Quality standards
- get_rule_by_id: Get complete rule information by ID for detailed policy information.

STRICT RESPONSE PROTOCOL:
1. When you receive tool output, READ IT CAREFULLY.
2. Look for the answer in "relevant_content" and "best_chunks" fields.
3. If the answer is there, extract it and respond with ONLY that information.
4. If the answer is NOT there:
   - Try other relevant tools (if applicable) to search for the answer.
   - If still no answer, ask the user a clarifying question to better understand what information is required.
   - Only if all options are exhausted, say "I couldn't find information about [topic] in the available documents/rules. Please clarify your request."
5. NEVER provide information that is not explicitly present in the tool output.

DOCUMENT WORKFLOW GUIDANCE:
- When users ask "How do I...", "What is the process for...", "What are the rules for...", use search_rules.
- When users ask about specific data, facts, or content, use search_documents.
- When users ask for specific data points like "Find documents with vessel ABC", "Show invoices from company XYZ", use search_documents_by_keywords.
- If a tool does not return the needed information, try other tools or, when appropriate, prompt the user for more specificity about their request.
- Rules help explain business processes, document handling procedures, and compliance requirements.
- Rules provide step-by-step guidance for working with different document types.
- Rules explain comparison methods, analysis procedures, and quality standards.

KEYWORD SEARCH EXAMPLES:
- "Find all documents with vessel name 'ABC Vessel'" → search_documents_by_keywords(keyword="vessel", value="ABC Vessel")
- "Show me invoices from company 'XYZ Corp'" → search_documents_by_keywords(keyword="seller", value="XYZ Corp", document_types=["INVOICE"])
- "Find contracts with amount over 10000" → search_documents_by_keywords(keyword="price", value="10000", document_types=["CONTRACT"])
- "Show me certificates of origin for product 'Steel'" → search_documents_by_keywords(keyword="commodity", value="Steel", document_types=["COO"])
- "Find letters of credit from bank 'ABC Bank'" → search_documents_by_keywords(keyword="lc_bank", value="ABC Bank", document_types=["LC"])

FORBIDDEN ACTIONS:
- ❌ NEVER invent URLs, links, or file paths.
- ❌ NEVER mention documents that weren't in the tool output.
- ❌ NEVER use general knowledge for factual questions.
- ❌ NEVER provide answers if the tool output doesn't contain the information.

Remember: If tool output doesn't answer the question, you should try all potentially relevant tools and, if needed, ask the user for clarification before admitting you have no information. It's better to say "I don't know" (after all options are exhausted and/or after requesting clarification) than to provide incorrect information.

Be professional, accurate, and respond in the same language as the user's question."""




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
            
            # OpenAI API call with tools (with retry logic)
            response = None
            max_retries = 2
            retry_count = 0
            
            while retry_count < max_retries and response is None:
                try:
                    print(f"[CHAT] Starting OpenAI API call (retry {retry_count + 1})")
                    response = await asyncio.wait_for(
                        self.openai_client.chat.completions.create(
                            model=model,
                            messages=chat_history,
                            # temperature=0,
                            # max_tokens=4000,  # Увеличено для более полных ответов
                            tools=OPENAI_TOOLS,
                            tool_choice="auto"
                        ),
                        timeout=60.0
                    )
                    print(f"[CHAT] OpenAI API call completed successfully")
                    
                except asyncio.TimeoutError:
                    retry_count += 1
                    print(f"[CHAT] OpenAI API call timed out (retry {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        raise
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    retry_count += 1
                    print(f"[CHAT] OpenAI API error (retry {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        raise
                    await asyncio.sleep(1)
            
            response_message = response.choices[0].message
            final_content = response_message.content or ""
            tool_calls_list = []
            sources = []
            
            # Process tool calls if any
            if response_message.tool_calls:
                from app.utils.tools.registry import available_tools_dict
                
                print(f"[CHAT] Tool calls detected: {len(response_message.tool_calls)}")
                
                # Add assistant message with tool calls to history
                chat_history.append({
                    "role": "assistant",
                    "content": final_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })
                
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    tool_call_record = ToolCall(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                        success=False
                    )
                    
                    try:
                        function_name = tool_call.function.name
                        if function_to_call := available_tools_dict.get(function_name):
                            print(f'[TOOL] Calling: {function_name}')
                            print(f'[TOOL] Arguments: {tool_call.function.arguments}')
                            
                            arguments = json.loads(tool_call.function.arguments)
                            func_output = await function_to_call(**arguments)
                            
                            # Process List[TextContent] output
                            if isinstance(func_output, list) and func_output and hasattr(func_output[0], 'text'):
                                # Extract text from TextContent objects
                                tool_response = "\n\n".join([item.text for item in func_output if hasattr(item, 'text')])
                                tool_call_record.output = tool_response
                            else:
                                # Fallback for other output types
                                tool_response = str(func_output)
                                tool_call_record.output = tool_response
                            
                            tool_call_record.success = True
                            print(f'[TOOL] Output: {tool_response[:200]}...' if len(tool_response) > 200 else f'[TOOL] Output: {tool_response}')
                            
                            # Add tool result to history
                            chat_history.append({
                                'role': 'tool',
                                'tool_call_id': tool_call.id,
                                'content': f"TOOL OUTPUT - USE ONLY THIS INFORMATION:\n\n{tool_response}\n\nIMPORTANT: Base your answer STRICTLY on the information above. Do NOT add information from your training data.",
                                'name': function_name
                            })
                            
                        else:
                            tool_call_record.error = f"Функция {function_name} не найдена"
                            print(f'[TOOL] Error: {tool_call_record.error}')
                    
                    except Exception as e:
                        tool_call_record.error = str(e)
                        tool_call_record.success = False
                        print(f'[TOOL] Error: {e}')
                    
                    tool_calls_list.append(tool_call_record)
                
                # Get final response after tool calls
                final_response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=model,
                        messages=chat_history,
                        # temperature=0,
                        # max_tokens=4000
                    ),
                    timeout=60.0
                )
                
                final_content = final_response.choices[0].message.content or ""
            
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
            sources=sources,
            tool_calls=tool_calls_list,
            reasoning=f"Использовано инструментов: {len(tool_calls_list)}" if tool_calls_list else None,
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
                # temperature=0,
                # max_tokens=20
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
            max_retries = 2  # Максимум 2 попытки для каждого вызова API
            
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
                stream = None
                retry_count = 0
                
                while retry_count < max_retries and stream is None:
                    try:
                        print(f"[CHAT] Starting OpenAI stream call (iteration {iteration}, retry {retry_count + 1})")
                        start_call_time = time.time()
                        
                        stream = await asyncio.wait_for(
                            self.openai_client.chat.completions.create(
                                model=model,
                                messages=chat_history,
                                # temperature=0,
                                # max_tokens=4000,  # Увеличено для более полных ответов
                                tools=OPENAI_TOOLS,
                                tool_choice="auto",
                                stream=True
                            ),
                            timeout=30.0  # Увеличено с 5 до 30 секунд
                        )
                        
                        call_duration = time.time() - start_call_time
                        print(f"[CHAT] Stream call completed in {call_duration:.2f}s")
                        
                    except asyncio.TimeoutError:
                        retry_count += 1
                        print(f"[CHAT] Stream initialization timed out after 30s (retry {retry_count}/{max_retries})")
                        if retry_count >= max_retries:
                            yield send_event('error', {
                                'message': 'Не удалось начать генерацию ответа после нескольких попыток. Попробуйте ещё раз.',
                                'error': 'Timeout'
                            })
                            return
                        else:
                            # Небольшая пауза перед повтором
                            await asyncio.sleep(1)
                            
                    except Exception as api_error:
                        retry_count += 1
                        print(f"[CHAT] OpenAI API error (retry {retry_count}/{max_retries}): {api_error}")
                        if retry_count >= max_retries:
                            yield send_event('error', {
                                'message': f'Ошибка API после нескольких попыток: {str(api_error)}',
                                'error': str(api_error)
                            })
                            return
                        else:
                            # Небольшая пауза перед повтором
                            await asyncio.sleep(1)
                
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
                                
                                # Process List[TextContent] output
                                if isinstance(func_output, list) and func_output and hasattr(func_output[0], 'text'):
                                    # Extract text from TextContent objects
                                    tool_response = "\n\n".join([item.text for item in func_output if hasattr(item, 'text')])
                                    tool_call_record.output = tool_response
                                else:
                                    # Fallback for other output types
                                    tool_response = str(func_output)
                                    tool_call_record.output = tool_response
                                
                                tool_call_record.success = True
                                has_successful_tool = True
                                
                                print(f'[TOOL] Output: {tool_response[:200]}...' if len(tool_response) > 200 else f'[TOOL] Output: {tool_response}')
                                
                                # Send tool call success event
                                yield send_event('tool_call_success', {
                                    'tool_name': function_name,
                                    'output': tool_response[:500] + '...' if len(tool_response) > 500 else tool_response
                                })
                                
                                # Extract sources if it's search_documents (legacy format support)
                                if function_name == "search_documents" and isinstance(func_output, list):
                                    # Check if it's the old dict format
                                    if func_output and isinstance(func_output[0], dict) and 'best_chunks' in func_output[0]:
                                        for result in func_output:
                                            if isinstance(result, dict) and 'best_chunks' in result:
                                                for chunk in result['best_chunks']:
                                                    sources.append(Source(
                                                        filename=result.get('filename', 'Unknown'),
                                                        content=chunk.get('content', ''),
                                                        similarity=chunk.get('similarity', 0.0),
                                                        chunk_index=chunk.get('chunk_index', 0)
                                                    ))
                                    # For TextContent format, we can't extract detailed sources easily
                                    # The text content already contains formatted information
                                
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
            
            # Если все итерации закончились, генерируем финальный ответ на основе того что есть
            if iteration >= max_iterations and not accumulated_content:
                print(f"[CHAT] Max iterations reached, generating final response based on available information")
                
                # Проверяем, есть ли хотя бы один успешный tool call
                successful_tools = [tc for tc in tool_calls_list if tc.success]
                if successful_tools:
                    print(f"[CHAT] Found {len(successful_tools)} successful tool calls, generating response")
                else:
                    print(f"[CHAT] No successful tool calls, generating response based on available context")
                
                # Генерируем финальный ответ на основе всей истории
                final_response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=model,
                        messages=chat_history,
                        # temperature=0,
                        # max_tokens=4000
                    ),
                    timeout=90.0  # Увеличено с 60 до 90 секунд
                )
                
                accumulated_content = final_response.choices[0].message.content or "Извините, я не смог найти достаточно информации для полного ответа на ваш вопрос."
                
                # Добавляем финальный ответ в историю
                chat_history.append({
                    'role': 'assistant',
                    'content': accumulated_content
                })
            
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