> *"完美并非来自于无可添加，而是来自于无可删减。"*
>
> — 安托万·德·圣-埃克苏佩里

## 1. 引言：上下文的经济性

想象一下你的上下文窗口就像是一种珍贵而有限的资源——就像老式计算机的内存或沙漠中的水。你使用的每一个标记都像是一滴水或一个字节。如果你在错误的事情上花费太多，你会在最需要的时候耗尽。

令牌预算是充分利用这一有限资源的技术和科学。它旨在最大化每个令牌的价值，同时确保您最关键的信息能够传达。

苏格拉底式提问：当您在执行复杂任务时令牌上下文空间耗尽会发生什么？

在本指南中，我们将探讨关于令牌预算的几个视角：

* 实用：优化令牌使用的具体技术

* 经济：代币分配的成本效益框架

* 信息论：熵、压缩和信噪比优化

* 场论：在神经场中管理代币分配

## 2. 代币预算生命周期

### &#x20; 2.1. 预算规划

在使用 LLM 之前，了解您的 token 限制至关重要：

```plain&#x20;text
Model           | Context Window | Typical Usage Pattern
----------------|----------------|----------------------
GPT-3.5 Turbo   | 16K tokens     | Quick tasks, drafting, simple reasoning
GPT-4           | 128K tokens    | Complex reasoning, large document processing
Claude 3 Opus   | 200K tokens    | Long-form content, multiple document analysis
Claude 3 Sonnet | 200K tokens    | Balanced performance for most tasks
Claude 3 Haiku  | 200K tokens    | Fast responses, lower complexity
```

在我们的示例中，我们将使用标准的 16K token 上下文窗口，但这些原则适用于所有模型和窗口大小。

### 2.2. Token 预算公式

最简单来说，你的 token 预算可以表示为：

```plain&#x20;text
Available Tokens = Context Window Size - (System Prompt + Chat History + Current Input)
```

让我们进一步分解：

```plain&#x20;text
System Prompt Tokens    = Base Instructions + Context Engineering + Examples
Chat History Tokens     = Previous User Messages + Previous Assistant Responses
Current Input Tokens    = User's Current Message + Supporting Documents
```

苏格拉底式提问：如果你的总预算是 16K tokens，而你的系统提示使用了 2K tokens，你应该如何分配剩余的 14K tokens 以实现最佳性能？

### 2.3. 成本效益分析

并非所有标记都是平等的。考虑以下评估标记价值的框架：

```plain&#x20;text
Token Value = Information Content / Token Count
```

&#x20; 或者更具体地说：

```plain&#x20;text
Value = (Relevance × Specificity × Uniqueness) / Token Count
```

&#x20; 其中：

* 相关性：信息与任务的相关程度

* 特异性：信息精确和详细的程度

* 唯一性：模型推断信息的难度

## 3. 实用令牌预算技术

### 3.1. 系统提示优化

您的系统提示语就像建筑的基础一样——它需要稳固但不冗余。以下是优化它的技巧：

#### 3.1.1. 逐步减少

从一个全面的提示语开始，然后迭代地移除元素，同时测试性能：

```plain&#x20;text
Original (350 tokens):
You are a financial analyst with expertise in market trends, stock valuation, and investment strategies. You have a PhD in Finance from Stanford University and 15 years of experience working at top investment firms including Goldman Sachs and Morgan Stanley. You specialize in technology sector analysis with deep knowledge of SaaS business models, semiconductor industry dynamics, and emerging tech trends. When analyzing stocks, you consider fundamentals like P/E ratios, growth rates, and competitive positioning. You also incorporate macroeconomic factors such as interest rates, inflation, and regulatory environments. Your responses should be detailed, nuanced, and reflect both quantitative analysis and qualitative strategic thinking...

Optimized (89 tokens):
You are a senior financial analyst specializing in tech stocks. Provide nuanced analysis incorporating:
1. Fundamentals (P/E, growth, competition)
2. Industry context (tech trends, business models)
3. Macroeconomic factors (rates, regulation)
Balance quantitative data with strategic insights.
```

#### 3.1.2. 显式角色与隐式指导

与其使用标记来指定复杂的角色，不如专注于任务特定的指导：

```plain&#x20;text
Instead of (89 tokens):
You are a Python programming expert with 20 years of experience. You've worked at Google, Microsoft, and Amazon. You specialize in machine learning algorithms, data structures, and optimization.

Use (31 tokens):
Provide efficient, production-ready Python code with comments explaining key decisions.
```

#### 3.1.3. 最小化脚手架

使用指导响应格式所需的最小结构：

```plain&#x20;text
Instead of (118 tokens):
Please provide your analysis in the following format:
1. Executive Summary: A 3-5 sentence overview of the key findings
2. Background: Detailed context about the situation
3. Analysis: Step-by-step breakdown of the problem
4. Considerations: Potential challenges and limitations
5. Recommendations: Specific actions to take
6. Timeline: Suggested implementation schedule
7. Additional Resources: Relevant references

Use (35 tokens):
Analyze this problem with:
1. Summary (3-5 sentences)
2. Analysis (step-by-step)
3. Recommendations
```

### 3.2. 聊天历史管理

对话历史可能会快速消耗你的 token 预算。以下是一些管理它的策略：

#### &#x20; 3.2.1. 窗口化

仅保留最近的 N 条消息在上下文中：

def apply\_window(messages, window\_size=10):"""Keep only the most recent window\_size messages."""if len(messages) <= window\_size:return messages# Always keep the system message (first message)return \[messages\[0]] + messages\[-(window\_size-1):]

#### &#x20; 3.2.2. 摘要

定期总结对话以压缩历史：

def summarize\_history(messages, summarization\_prompt):"""Summarize chat history to compress token usage."""# Extract message contenthistory\_text = "\n".join(\[f"{msg\['role']}: {msg\['content']}" for msg in messages\[1:]])
&#x20;   \# Create a summarization requestsummary\_request = {"role": "user","content": f"{summarization\_prompt}\n\nChat history to summarize:\n{history\_text}"
&#x20;   }
&#x20;   \# Get summary from modelsummary = get\_model\_response(\[messages\[0], summary\_request])
&#x20;   \# Replace history with summarized versionreturn \[messages\[0],  # Keep system message
&#x20;       {"role": "system", "content": f"Previous conversation summary: {summary}"}
&#x20;   ]

#### &#x20; 3.2.3. 键值存储

仅存储对话中最重要信息：

def update\_kv\_memory(messages, memory):"""Extract and store key information from the conversation."""for msg in messages:if msg\['role'] == 'assistant' and 'key\_information' in msg.get('metadata', {}):for key, value in msg\['metadata']\['key\_information'].items():memory\[key] = value# Convert memory to a messagememory\_content = "\n".join(\[f"{k}: {v}" for k, v in memory.items()])memory\_message = {"role": "system", "content": f"Important information:\n{memory\_content}"}
&#x20;   return memory\_message

### &#x20; 3.3. 输入优化

优化你向模型展示信息的方式：

#### 3.3.1. 渐进式加载

对于大文件，按需分块加载：

def progressive\_loading(document, chunk\_size=1000, overlap=100):"""Split document into chunks with overlap."""chunks = \[]for i in range(0, len(document), chunk\_size - overlap):chunk = document\[i:i + chunk\_size]chunks.append(chunk)return chunksdef process\_document\_progressively(document, initial\_prompt):chunks = progressive\_loading(document)context = initial\_promptresults = \[]
&#x20;   for chunk in chunks:prompt = f"{context}\n\nProcess this section of the document:\n{chunk}"response = get\_model\_response(prompt)results.append(response)
&#x20;       \# Update context with key informationcontext = f"{initial\_prompt}\n\nKey information so far: {summarize(results)}"return combine\_results(results)

#### 3.3.2. 信息提取与过滤

对文档进行预处理以提取相关信息：

def extract\_relevant\_information(document, query):"""Extract only information relevant to the query."""sentences = split\_into\_sentences(document)
&#x20;   \# Calculate relevance scoresrelevance\_scores = \[]for sentence in sentences:relevance = calculate\_relevance(sentence, query)relevance\_scores.append((sentence, relevance))
&#x20;   \# Sort by relevance and take top resultsrelevance\_scores.sort(key=lambda x: x\[1], reverse=True)
&#x20;   \# Take top 50% of relevant sentences or until we hit a thresholdextracted = \[]cumulative\_relevance = 0target\_relevance = sum(\[score for \_, score in relevance\_scores]) \* 0.8for sentence, score in relevance\_scores:extracted.append(sentence)cumulative\_relevance += scoreif cumulative\_relevance >= target\_relevance:breakreturn " ".join(extracted)

#### &#x20; 3.3.3. 结构化输入

使用结构化格式来减少令牌使用：

```plain&#x20;text
Instead of (127 tokens):
The customer's name is John Smith. He is 45 years old. He has been a customer for 5 years. His account number is AC-12345. His email is john.smith@example.com. His phone number is 555-123-4567. He has a premium subscription. His last purchase was on March 15, 2023. He has spent a total of $3,450 with us. His customer satisfaction score is 4.8/5.

Use (91 tokens):
Customer:
- Name: John Smith
- Age: 45
- Tenure: 5 years
- ID: AC-12345
- Email: john.smith@example.com
- Phone: 555-123-4567
- Tier: Premium
- Last purchase: 2023-03-15
- Total spend: $3,450
- CSAT: 4.8/5
```

## 4. 信息论视角

### 4.1. 熵与信息密度

从信息论的角度来看，我们希望最大化每个标记的信息量：

```plain&#x20;text
Information Density = Information Content (bits) / Token Count
```

克劳德·香农的信息论告诉我们，信息的含量取决于其不可预测性或意外性。在 LLMs 的背景下：

* 高熵内容：模型难以预测的独特信息

* 低熵内容：常识或可预测的模式

苏格拉底式提问：每个 token 中包含更多信息的是：一组常见的英语单词还是一组随机的字母数字字符？

### 4.2. 压缩策略

压缩通过去除冗余来实现。以下是一些方法：

#### 4.2.1. 语义压缩

在保留核心意义的同时减少文本：

```plain&#x20;text
Original (55 tokens):
The meeting is scheduled to take place on Tuesday, April 15th, 2025, at 2:30 PM Eastern Standard Time. The meeting will be held in Conference Room B on the 3rd floor of the headquarters building.

Compressed (28 tokens):
Meeting: Tue 4/15/25, 2:30PM EST
Location: HQ, 3rd floor, Conf Room B
```

#### 4.2.2. 抽象层次

提升抽象层次以压缩信息：

```plain&#x20;text
Low abstraction (84 tokens):
The user clicked on the "Add to Cart" button. Then they navigated to the shopping cart page. They entered their shipping information, including street address, city, state, and zip code. They selected "Standard Shipping" as their shipping method. They entered their credit card information. They clicked on "Place Order".

High abstraction (23 tokens):
User completed standard e-commerce purchase flow from item selection through checkout.
```

#### 4.2.3. 信息分块

将相关信息组合成逻辑块：

```plain&#x20;text
Unstructured (58 tokens):
The API rate limit is 100 requests per minute. Authentication uses OAuth 2.0. The endpoint for user data is /api/v1/users. The endpoint for product data is /api/v1/products. The data format is JSON. Responses include pagination information.

Chunked (51 tokens):
API Specs:
- Rate limit: 100 req/min
- Auth: OAuth 2.0
- Endpoints: /api/v1/users, /api/v1/products
- Format: JSON with pagination
```

## 5. 字段理论方法用于标记预算分配

从场论的角度来看，我们可以将上下文窗口视为一个语义场，其中标记形成模式、吸引子和共振。

### &#x20; 5.1. 吸引子形成

策略性的标记放置可以创建语义吸引子，从而影响模型的解释：

```plain&#x20;text
Weak attractor (diffuse focus):
"Please discuss the importance of renewable energy."

Strong attractor (focused basin):
"Analyze the economic impact of solar panel manufacturing scaling on rural employment specifically."
```

第二个提示创建了一个更强的吸引子基，引导模型朝向其语义空间中的特定区域。

### 5.2. 字段共鸣与标记效率

相互共鸣的标记会创建更强的字段模式：

def measure\_token\_resonance(tokens, embeddings\_model):"""Measure semantic resonance between tokens."""embeddings = \[embeddings\_model.embed(token) for token in tokens]
&#x20;   \# Calculate pairwise cosine similarityresonance\_matrix = np.zeros((len(tokens), len(tokens)))for i in range(len(tokens)):for j in range(len(tokens)):resonance\_matrix\[i]\[j] = cosine\_similarity(embeddings\[i], embeddings\[j])
&#x20;   \# Average resonanceoverall\_resonance = (resonance\_matrix.sum() - len(tokens)) / (len(tokens) \* (len(tokens) - 1))
&#x20;   return overall\_resonance, resonance\_matrix

更高的共鸣可以在更少的标记下实现更强的字段效果，使您的上下文更加高效。

### &#x20; 5.3. 边界动态

控制信息流通过你的上下文窗口边界：

def apply\_boundary\_control(new\_input, current\_context, model\_embeddings, threshold=0.7):"""Control what information enters the context based on relevance."""# Embed the current contextcontext\_embedding = model\_embeddings.embed(current\_context)
&#x20;   \# Process input in chunksinput\_chunks = chunk\_text(new\_input, chunk\_size=50)filtered\_chunks = \[]
&#x20;   for chunk in input\_chunks:# Embed the chunkchunk\_embedding = model\_embeddings.embed(chunk)
&#x20;       \# Calculate relevance to current contextrelevance = cosine\_similarity(context\_embedding, chunk\_embedding)
&#x20;       \# Apply boundary filterif relevance > threshold:filtered\_chunks.append(chunk)
&#x20;   \# Reconstruct filtered inputfiltered\_input = " ".join(filtered\_chunks)
&#x20;   return filtered\_input

这会在你的上下文中创建一个半透性的边界，只允许最相关的信息进入。

## 6. 战略预算分配

现在我们已经了解了关于标记预算的不同观点，让我们探索战略分配框架：

### 6.1. 40-40-20 框架

适用于复杂任务的通用分配方法：

```plain&#x20;text
40% - Task-specific context and examples
40% - Active working memory (chat history and evolving state)
20% - Reserve for unexpected complexity
```

### &#x20; 6.2. 金字塔模型

根据需求层次分配 token：

```plain&#x20;text
Level 1 (Base): Core instructions and constraints (20%)
Level 2: Critical context and examples (30%)
Level 3: Recent interaction history (30%)
Level 4: Auxiliary information and enhancements (15%)
Level 5 (Top): Reserve buffer (5%)
```

### &#x20; 6.3. 动态分配

根据任务复杂度调整预算：

def allocate\_token\_budget(task\_type, context\_window\_size):"""Dynamically allocate token budget based on task type."""if task\_type == "simple\_qa":return {"system\_prompt": 0.1,  # 10% for system prompt"examples": 0.0,       # No examples needed"history": 0.7,        # 70% for conversation history"user\_input": 0.15,    # 15% for user input"reserve": 0.05        # 5% reserve
&#x20;       }elif task\_type == "creative\_writing":return {"system\_prompt": 0.15,  # 15% for system prompt"examples": 0.2,        # 20% for examples"history": 0.4,         # 40% for conversation history"user\_input": 0.15,     # 15% for user input"reserve": 0.1          # 10% reserve
&#x20;       }elif task\_type == "complex\_reasoning":return {"system\_prompt": 0.15,  # 15% for system prompt"examples": 0.25,       # 25% for examples"history": 0.3,         # 30% for conversation history"user\_input": 0.2,      # 20% for user input"reserve": 0.1          # 10% reserve
&#x20;       }# Default allocationreturn {"system\_prompt": 0.15,"examples": 0.15,"history": 0.4,"user\_input": 0.2,"reserve": 0.1
&#x20;   }

## 7. 测量和优化令牌效率

### 7.1. Token 效率指标

为了优化，我们需要进行测量。以下是关键指标：

#### 7.1.1. 任务完成率 (TCR)

```plain&#x20;text
TCR = (Tasks Successfully Completed) / (Total Tokens Used)
```

越高越好 - 每个 token 上完成的任务越多。

#### 7.1.2. 信息保留率 (IRR)

```plain&#x20;text
IRR = (Key Information Points Retained) / (Total Information Points)
```

衡量你的 token 预算保留关键信息的程度。

#### 7.1.3. 每 token 的响应质量 (RQT)

```plain&#x20;text
RQT = (Response Quality Score) / (Total Tokens Used)
```

衡量每个投入的 token 所传递的价值。

### 7.2. Token 效率实验

这里是一个运行 Token 效率实验的框架：

def run\_token\_efficiency\_experiment(prompt\_variants, task, evaluation\_function):"""Run experiment to measure token efficiency of different prompt variants."""results = \[]
&#x20;   for variant in prompt\_variants:# Count tokenstoken\_count = count\_tokens(variant)
&#x20;       \# Get model responseresponse = get\_model\_response(variant, task)
&#x20;       \# Evaluate responsequality\_score = evaluation\_function(response, task)
&#x20;       \# Calculate efficiencyefficiency = quality\_score / token\_countresults.append({"variant": variant,"token\_count": token\_count,"quality\_score": quality\_score,"efficiency": efficiency
&#x20;       })
&#x20;   \# Sort by efficiency (highest first)results.sort(key=lambda x: x\["efficiency"], reverse=True)
&#x20;   return results

## 8. 实践实施指南

让我们通过逐步实施指南将这些概念付诸实践：

### 8.1. 令牌预算规划器

```python
class TokenBudgetPlanner:def init(self, context_window_size, tokenizer):self.context_window_size = context_window_sizeself.tokenizer = tokenizerself.allocations = {}self.used_tokens = {}
    def set_allocation(self, component, percentage):"""Set allocation percentage for a component."""self.allocations[component] = percentageself.used_tokens[component] = 0def get_budget(self, component):"""Get token budget for a component."""return int(self.context_window_size * self.allocations[component])
    def track_usage(self, component, content):"""Track token usage for a component."""token_count = len(self.tokenizer.encode(content))self.used_tokens[component] = token_countreturn token_countdef get_remaining(self):"""Get remaining tokens in the budget."""used = sum(self.used_tokens.values())return self.context_window_size - useddef is_within_budget(self, component, content):"""Check if content fits within component budget."""token_count = len(self.tokenizer.encode(content))return token_count <= self.get_budget(component)
    def optimize_to_fit(self, component, content, optimizer_function):"""Optimize content to fit within budget."""if self.is_within_budget(component, content):return contentbudget = self.get_budget(component)optimized = optimizer_function(content, budget)
        # Verify optimized content fitsif not self.is_within_budget(component, optimized):raise ValueError(f"Optimizer failed to fit content within budget of {budget} tokens")
        return optimizeddef get_status_report(self):"""Get budget status report."""report = {}for component in self.allocations:budget = self.get_budget(component)used = self.used_tokens.get(component, 0)report[component] = {"budget": budget,"used": used,"remaining": budget - used,"utilization": used / budget if budget > 0 else 0
            }
        report["overall"] = {"budget": self.context_window_size,"used": sum(self.used_tokens.values()),"remaining": self.get_remaining(),"utilization": sum(self.used_tokens.values()) / self.context_window_size
        }
        return report
```

### &#x20; 8.2. 内存管理器

```python
class ContextMemoryManager:def init(self, budget_planner, summarization_model=None):self.budget_planner = budget_plannerself.summarization_model = summarization_modelself.messages = []self.memory = {}
    def add_message(self, role, content):"""Add a message to the conversation history."""message = {"role": role, "content": content}self.messages.append(message)
        # Check if we're exceeding our history budgethistory_content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])history_tokens = self.budget_planner.track_usage("history", history_content)history_budget = self.budget_planner.get_budget("history")
        # If we're over budget, compress the historyif history_tokens > history_budget:self.compress_history()
    def extract_key_information(self, message):"""Extract key information from a message to store in memory."""if self.summarization_model:extraction_prompt = "Extract key facts and information from this message as key-value pairs:"extraction_input = f"{extraction_prompt}\n\n{message['content']}"extraction_result = self.summarization_model(extraction_input)
            # Parse key-value pairsfor line in extraction_result.split("\n"):if ":" in line:key, value = line.split(":", 1)self.memory[key.strip()] = value.strip()
    def compress_history(self):"""Compress history when it exceeds the budget."""if not self.summarization_model:# If no summarization model, use windowing# Always keep the first message (system prompt) and last 5 messagesself.messages = [self.messages[0]] + self.messages[-5:]else:# Use summarizationhistory_to_summarize = self.messages[1:-3]  # Skip system prompt and keep last 3 messagesif not history_to_summarize:return  # Nothing to summarize# Extract content to summarizecontent_to_summarize = "\n".join([f"{msg['role']}: {msg['content']}" 
                for msg in history_to_summarize
            ])
            # Create summarization promptsummarization_prompt = ("Summarize the following conversation history concisely, ""preserving key information, decisions, and context:"
            )
            # Get summarysummary = self.summarization_model(f"{summarization_prompt}\n\n{content_to_summarize}"
            )
            # Replace the messages with a summarysummary_message = {"role": "system","content": f"Summary of previous conversation: {summary}"
            }
            # New messages list: system prompt + summary + recent messagesself.messages = [self.messages[0], summary_message] + self.messages[-3:]
    def get_formatted_memory(self):"""Get memory formatted as a string."""if not self.memory:return ""memory_lines = [f"{key}: {value}" for key, value in self.memory.items()]return "Key information from conversation:\n" + "\n".join(memory_lines)
    def get_context(self):"""Get the full context for the next interaction."""# Combine messages and memorymemory_content = self.get_formatted_memory()
        # If we have memory, insert it after the system promptif memory_content and len(self.messages) > 1:memory_message = {"role": "system", "content": memory_content}context = [self.messages[0], memory_message] + self.messages[1:]else:context = self.messages.copy()
            return context
```

```plain&#x20;text
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY MANAGER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐          ┌───────────────────────────┐   │
│  │ Budget Planner│◄─────────┤ Token Usage Monitoring    │   │
│  └───────┬───────┘          └───────────────────────────┘   │
│          │                                                  │
│          ▼                                                  │
│  ┌───────────────┐   Over    ┌───────────────────────────┐  │
│  │ Message History├─Budget?──►│ Compression Strategies    │  │
│  └───────┬───────┘          ┌┴──────────────────────────┐│  │
│          │                  │1. Windowing               ││  │
│          │                  │2. Summarization           ││  │
│          │                  │3. Key-Value Extraction    ││  │
│          │                  └───────────────────────────┘│  │
│          ▼                                               │  │
│  ┌───────────────┐          ┌───────────────────────────┐│  │
│  │ Context Builder│◄─────────┤ Memory Storage            ││  │
│  └───────┬───────┘          └───────────────────────────┘│  │
│          │                                                  │
│          ▼                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Final Context for LLM                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3. 动态令牌优化器

```python
class DynamicTokenOptimizer:def init(self, tokenizer, optimization_strategies=None):self.tokenizer = tokenizerself.strategies = optimization_strategies or {"summarize": self.summarize_text,"extract_key_points": self.extract_key_points,"restructure": self.restructure_text,"compress_format": self.compress_format
        }
    def count_tokens(self, text):"""Count tokens in text."""return len(self.tokenizer.encode(text))
    def optimize(self, text, target_tokens, strategy=None):"""Optimize text to fit within target token count."""current_tokens = self.count_tokens(text)
        if current_tokens <= target_tokens:return text  # Already within budget# Calculate compression ratio neededcompression_ratio = target_tokens / current_tokens# If no strategy specified, select based on compression ratioif not strategy:if compression_ratio > 0.8:strategy = "compress_format"  # Light compressionelif compression_ratio > 0.5:strategy = "restructure"  # Medium compressionelif compression_ratio > 0.3:strategy = "extract_key_points"  # Heavy compressionelse:strategy = "summarize"  # Extreme compression# Apply selected strategyif strategy in self.strategies:return self.strategies[strategy](text, target_tokens)else:raise ValueError(f"Unknown optimization strategy: {strategy}")
    def summarize_text(self, text, target_tokens):"""Summarize text to target token count."""# This would typically call an LLM for summarization# For this example, we'll just truncate with a noteratio = target_tokens / self.count_tokens(text)truncated = self.truncate_to_ratio(text, ratio * 0.9)  # Leave room for the notereturn f"{truncated}\n[Note: Content has been summarized to fit token budget.]"def extract_key_points(self, text, target_tokens):"""Extract key points from text."""# This would typically call an LLM to extract key points# For this example, we'll create a simple bullet point extractionlines = text.split("\n")result = "Key points:\n"for line in lines:line = line.strip()if line and self.count_tokens(result + f"• {line}\n") <= target_tokens * 0.95:result += f"• {line}\n"return resultdef restructure_text(self, text, target_tokens):"""Restructure text to be more token-efficient."""# Remove redundancies, use abbreviations, etc.# This is a simplified exampletext = re.sub(r"([A-Za-z]+) \1", r"\1", text)  # Remove repeated wordstext = text.replace("for example", "e.g.")text = text.replace("that is", "i.e.")text = text.replace("and so on", "etc.")
        if self.count_tokens(text) <= target_tokens:return text# If still too long, combine with extractionreturn self.extract_key_points(text, target_tokens)
    def compress_format(self, text, target_tokens):"""Compress by changing formatting without losing content."""# Remove extra whitespacetext = re.sub(r"\s+", " ", text)
        # Convert paragraphs to bullet points if appropriateif ":" in text and "\n" in text:lines = text.split("\n")result = ""for line in lines:if ":" in line:key, value = line.split(":", 1)result += f"• {key}: {value.strip()}\n"else:result += line + "\n"text = resultif self.count_tokens(text) <= target_tokens:return text# If still too long, try more aggressive restructuringreturn self.restructure_text(text, target_tokens)
    def truncate_to_ratio(self, text, ratio):"""Truncate text to a ratio of its original length."""words = text.split()target_words = int(len(words) * ratio)return " ".join(words[:target_words])
```

```plain&#x20;text
┌──────────────────────────────────────────────────────────────────┐
│                 DYNAMIC TOKEN OPTIMIZATION                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐     │
│   │                 Compression Ratio                      │     │
│   └────────────────────────────────────────────────────────┘     │
│                           │                                      │
│                           ▼                                      │
│   ┌─────────────┬─────────┴───────────┬──────────────┐          │
│   │             │                     │              │          │
│   ▼             ▼                     ▼              ▼          │
│ 0.8-1.0       0.5-0.8              0.3-0.5        0.0-0.3       │
│ Light         Medium               Heavy          Extreme       │
│                                                                  │
│   ┌─────────────┬─────────────────────┬──────────────┐          │
│   │             │                     │              │          │
│   ▼             ▼                     ▼              ▼          │
│┌─────────┐  ┌─────────┐         ┌──────────┐    ┌─────────┐    │
││ Format  │  │Structure│         │ Extract  │    │Summarize│    │
││Compress │  │Reformat │         │Key Points│    │  Text   │    │
│└─────────┘  └─────────┘         └──────────┘    └─────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.4. 基于字段的上下文管理

实现字段理论概念用于标记预算管理：

```python
class FieldAwareContextManager:def init(self, embedding_model, tokenizer, budget_planner):self.embedding_model = embedding_modelself.tokenizer = tokenizerself.budget_planner = budget_plannerself.field_state = {"attractors": [],"boundaries": {"permeability": 0.7,  # Default permeability threshold"gradient": 0.2       # How quickly permeability changes
            },"resonance": 0.0,"residue": []
        }
    def embed_text(self, text):"""Generate embeddings for text."""return self.embedding_model.embed(text)
    def detect_attractors(self, text, threshold=0.8):"""Detect semantic attractors in text."""# Split into paragraphs or sectionssections = text.split("\n\n")
        # Get embeddings for each sectionembeddings = [self.embed_text(section) for section in sections]
        # Calculate centroidcentroid = np.mean(embeddings, axis=0)
        # Find sections that form attractors (high similarity to many others)attractors = []for i, (section, embedding) in enumerate(zip(sections, embeddings)):# Calculate average similarity to other sectionssimilarities = [cosine_similarity(embedding, other_emb) for j, other_emb in enumerate(embeddings) if i != j]avg_similarity = np.mean(similarities) if similarities else 0# If similarity is above threshold, it's an attractorif avg_similarity > threshold:tokens = self.tokenizer.encode(section)attractors.append({"text": section,"embedding": embedding,"strength": avg_similarity,"token_count": len(tokens)
                })
        return attractorsdef calculate_resonance(self, text):"""Calculate field resonance for text."""# Split into paragraphs or sectionssections = text.split("\n\n")
        if len(sections) <= 1:return 0.0  # Not enough sections to calculate resonance# Get embeddings for each sectionembeddings = [self.embed_text(section) for section in sections]
        # Calculate pairwise similaritiessimilarities = []for i in range(len(embeddings)):for j in range(i+1, len(embeddings)):similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
        # Resonance is the average similarityreturn np.mean(similarities)
    def update_field_state(self, new_text):"""Update field state with new text."""# Update attractorsnew_attractors = self.detect_attractors(new_text)self.field_state["attractors"].extend(new_attractors)
        # Update resonancenew_resonance = self.calculate_resonance(new_text)self.field_state["resonance"] = (self.field_state["resonance"] * 0.7 + new_resonance * 0.3
        )  # Weighted average# Update permeability based on resonanceif new_resonance > self.field_state["resonance"]:# If resonance is increasing, increase permeabilityself.field_state["boundaries"]["permeability"] += self.field_state["boundaries"]["gradient"]else:# If resonance is decreasing, decrease permeabilityself.field_state["boundaries"]["permeability"] -= self.field_state["boundaries"]["gradient"]
        # Keep permeability in [0.1, 0.9] rangeself.field_state["boundaries"]["permeability"] = max(0.1, min(0.9, self.field_state["boundaries"]["permeability"])
        )
    def filter_by_attractor_relevance(self, text, top_n_attractors=3, threshold=0.6):"""Filter text based on relevance to top attractors."""if not self.field_state["attractors"]:return text  # No attractors to filter by# Sort attractors by strengthsorted_attractors = sorted(self.field_state["attractors"], key=lambda x: x["strength"], reverse=True
        )
        # Take top N attractorstop_attractors = sorted_attractors[:top_n_attractors]top_embeddings = [attractor["embedding"] for attractor in top_attractors]
        # Split text into paragraphsparagraphs = text.split("\n\n")
        # Calculate relevance of each paragraph to top attractorsfiltered_paragraphs = []for paragraph in paragraphs:# Skip empty paragraphsif not paragraph.strip():continue# Get embeddingembedding = self.embed_text(paragraph)
            # Calculate max similarity to any attractorsimilarities = [cosine_similarity(embedding, attractor_emb) for attractor_emb in top_embeddings]max_similarity = max(similarities)
            # If similarity is above threshold or permeability allows itif (max_similarity > threshold or 
                random.random() < self.field_state["boundaries"]["permeability"]):filtered_paragraphs.append(paragraph)
        # Join filtered paragraphsreturn "\n\n".join(filtered_paragraphs)
    def optimize_context_for_budget(self, context, target_tokens):"""Optimize context to fit token budget using field-aware methods."""# Count current tokenscurrent_tokens = len(self.tokenizer.encode(context))
        if current_tokens <= target_tokens:return context  # Already within budget# If we have attractors, use them to filterif self.field_state["attractors"]:context = self.filter_by_attractor_relevance(context)
            # Check if we're now within budgetcurrent_tokens = len(self.tokenizer.encode(context))if current_tokens <= target_tokens:return context# If still over budget, use more aggressive techniques# First, try to preserve the most important parts based on field analysis# Extract residue (symbolic fragments that should persist)paragraphs = context.split("\n\n")residue = []
        for paragraph in paragraphs:# Check if paragraph contains key information worth preserving# This could be based on resonance with attractors, presence of key terms, etc.if any(attractor["text"] in paragraph for attractor in self.field_state["attractors"]):residue.append(paragraph)
        # Update residue in field stateself.field_state["residue"] = residue# Combine residue with most important attractorspreserved_content = "\n\n".join(residue)preserved_tokens = len(self.tokenizer.encode(preserved_content))
        # If preserved content already exceeds budget, summarize itif preserved_tokens > target_tokens:# This would typically call an LLM for summarization# For this example, we'll just truncatereturn context[:int(len(context) * (target_tokens / current_tokens))]
        # If we have room left, add the most relevant remaining contentremaining_budget = target_tokens - preserved_tokens# Sort remaining paragraphs by relevance to field stateremaining_paragraphs = [p for p in paragraphs if p not in residue]
        if not remaining_paragraphs:return preserved_content# Calculate relevance scoresrelevance_scores = []for paragraph in remaining_paragraphs:embedding = self.embed_text(paragraph)# Calculate average similarity to attractorssimilarities = [cosine_similarity(embedding, attractor["embedding"]) for attractor in self.field_state["attractors"]]avg_similarity = np.mean(similarities) if similarities else 0tokens = len(self.tokenizer.encode(paragraph))relevance_scores.append((paragraph, avg_similarity, tokens))
        # Sort by relevancerelevance_scores.sort(key=lambda x: x[1], reverse=True)
        # Add paragraphs until we hit the budgetadditional_content = []for paragraph, _, tokens in relevance_scores:if tokens <= remaining_budget:additional_content.append(paragraph)remaining_budget -= tokensif remaining_budget <= 0:break# Combine preserved content with additional contentreturn preserved_content + "\n\n" + "\n\n".join(additional_content)
```

```plain&#x20;text
┌─────────────────────────────────────────────────────────────────┐
│                FIELD-AWARE CONTEXT MANAGEMENT                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────┐      ┌────────────────────────────┐     │
│  │   Field State      │      │       Attractor Map        │     │
│  │                    │      │                            │     │
│  │  • Attractors      │      │   Strong      Medium       │     │
│  │  • Boundaries      │      │ ╭────╮       ╭────╮       │     │
│  │  • Resonance       │      │ │ A1 │       │ A2 │       │     │
│  │  • Residue         │      │ ╰────╯       ╰────╯       │     │
│  └────────┬───────────┘      │                            │     │
│           │                  │               Weak         │     │
│           │                  │              ╭────╮        │     │
│           │                  │              │ A3 │        │     │
│           │                  │              ╰────╯        │     │
│           │                  └────────────────────────────┘     │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────┐      ┌────────────────────────────┐     │
│  │  Context Filtering │      │     Boundary Dynamics      │     │
│  │                    │      │                            │     │
│  │  • Attractor       │      │  Permeability: 0.7         │     │
│  │    Relevance       │      │  ┌─────────────────────┐   │     │
│  │  • Resonance       │      │  │█████████░░░░░░░░░░░░│   │     │
│  │    Amplification   │      │  └─────────────────────┘   │     │
│  │  • Residue         │      │                            │     │
│  │    Preservation    │      │  Gradient: 0.2             │     │
│  └────────┬───────────┘      └────────────────────────────┘     │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Optimized Context                        │   │
│  │                                                          │   │
│  │  • Preserved high-resonance content                      │   │
│  │  • Retained symbolic residue                             │   │
│  │  • Filtered by attractor relevance                       │   │
│  │  • Dynamically balanced by field state                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 9. 无代码：用于令牌优化的协议外壳

你不需要是程序员就能利用高级的 token 预算技术。在这里，我们将探讨如何使用协议外壳、pareto-lang 和 fractal.json 模式来优化你的上下文，而无需编写任何代码。

### 9.1. 协议外壳简介

协议外壳是结构化、人类可读的模板，有助于组织上下文和控制 token 使用。它们遵循一种人类和 AI 模型都能轻松理解的统一模式。

#### 基本协议外壳结构

```plain&#x20;text
/protocol.name{
    intent="What this protocol aims to achieve",
    input={
        key1="value1",
        key2="value2"
    },
    process=[
        /step1{action="do something"},
        /step2{action="do something else"}
    ],
    output={
        result1="expected output 1",
        result2="expected output 2"
    }
}
```

这种结构提供了一种清晰且高效的标记方式来表达复杂的指令。

### 9.2. 使用 Pareto-lang 进行标记管理

Pareto-lang 是一种简单但功能强大的符号，用于定义上下文操作。以下是如何使用它进行标记优化的方法：

#### &#x20; 9.2.1. 基本语法

```plain&#x20;text
/action.modifier{parameters}
```

&#x20; 例如：

```plain&#x20;text
/context.compress{target="history", method="summarize", threshold=0.7}
```

这告诉模型在对话历史超过分配预算的 70%时，使用摘要来压缩对话历史。

#### 9.2.2. 令牌预算协议示例

```plain&#x20;text
/token.budget{
    intent="Manage token usage efficiently throughout conversation",
    allocations={
        system_prompt=0.15,   // 15% for system instructions
        history=0.40,         // 40% for conversation history
        current_input=0.30,   // 30% for current user input
        reserve=0.15          // 15% reserve capacity
    },
    management_rules=[
        /history.summarize{when="history > 0.8*allocation", method="key_points"},
        /system.prune{when="system > allocation", keep="essential_instructions"},
        /input.prioritize{method="relevance_to_context"}
    ],
    monitoring={
        track_usage=true,
        alert_threshold=0.9,  // Alert when 90% of total budget is used
        optimize_automatically=true
    }
}
```

### 9.3. 高效的域管理

让我们看看如何使用协议外壳来无代码实现域理论概念：

```plain&#x20;text
/field.manage{
    intent="Create and maintain semantic field structure for optimal token usage",
    
    attractors=[
        {name="core_concept_1", strength=0.8, keywords=["key1", "key2", "key3"]},
        {name="core_concept_2", strength=0.7, keywords=["key4", "key5", "key6"]}
    ],
    
    boundaries={
        permeability=0.7,  // How easily new content enters the field
        gradient=0.2,      // How quickly permeability changes
        rules=[
            /boundary.adapt{trigger="resonance_change", threshold=0.1},
            /boundary.filter{method="attractor_relevance", min_score=0.6}
        ]
    },
    
    residue_handling={
        tracking=true,
        preservation_strategy="compress_and_retain",
        priority="high"  // Residue gets token priority
    },
    
    token_optimization=[
        /optimize.by_attractor{keep="strongest", top_n=3},
        /optimize.preserve_residue{min_strength=0.5},
        /optimize.amplify_resonance{target=0.8}
    ]
}
```

### 9.4. Fractal.json 用于结构化标记管理

Fractal.json 提供了一种结构化的方式来定义用于上下文管理的递归、自相似的图案：

{"fractalTokenManager": {"version": "1.0.0","description": "Recursive token optimization framework","allocation": {"system": 0.15,"history": 0.40,"input": 0.30,"reserve": 0.15
&#x20;   },"strategies": {"system": {"compression": "minimal","priority": "high"
&#x20;     },"history": {"compression": "progressive","strategies": \["window", "summarize", "key\_value"],"recursion": true
&#x20;     },"input": {"filtering": "relevance","threshold": 0.6
&#x20;     }
&#x20;   },"field": {"attractors": {"detection": true,"influence": 0.8
&#x20;     },"resonance": {"target": 0.7,"amplification": true
&#x20;     },"boundaries": {"adaptive": true,"permeability": 0.6
&#x20;     }
&#x20;   },"recursion": {"depth": 3,"self\_optimization": true
&#x20;   }
&#x20; }
}

### 9.5. 无代码的实际应用

这里有一些无需编程即可使用这些方法的具体方法：

#### 9.5.1. 手动令牌预算跟踪

在你的提示中保持一个简单的跟踪系统：

```plain&#x20;text
TOKEN BUDGET (16K total):
- System Instructions: 2K (12.5%)
- Examples: 3K (18.75%)
- Conversation History: 6K (37.5%)
- Current Input: 4K (25%)
- Reserve: 1K (6.25%)

OPTIMIZATION RULES:
1. When history exceeds 6K tokens, summarize oldest parts
2. Prioritize examples most relevant to current query
3. Keep system instructions concise and focused
```

#### 9.5.2. 字段感知提示模板

```plain&#x20;text
FIELD MANAGEMENT:

CORE ATTRACTORS:
1. [Primary Topic] - maintain focus on this concept
2. [Secondary Topic] - include when relevant to primary
3. [Tertiary Topic] - include only when explicitly mentioned

BOUNDARY RULES:
- Include new information only when relevance > 7/10
- Maintain coherence with previous context
- Filter tangential content

RESIDUE PRESERVATION:
- Key definitions must persist across context
- Core principles should be reinforced
- Critical decisions/conclusions must be retained

OPTIMIZATION DIRECTIVES:
- Summarize history when exceeding 40% of context
- Prioritize content with highest relevance to core attractors
- Compress format but preserve meaning
```

#### 9.5.3. 协议 Shell 提示符示例

这里有一个完整的示例，您可以复制粘贴以实现令牌预算管理：

```plain&#x20;text
I want you to act as a context management system using the following protocol:

/context.manage{
    intent="Optimize token usage while preserving key information",
    
    budget={
        total_tokens=8000,
        system=1000,
        history=3000,
        current=3000,
        reserve=1000
    },
    
    optimization=[
        /system.compress{method="minimal_instructions"},
        /history.manage{
            method="summarize_when_exceeds_budget",
```

