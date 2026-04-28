# 谁是卧底（LangChain + LangGraph + Streamlit）

Datawhale《easy-langent》**第八章**综合实践。在原有功能的基础上做了一些增强：多智能体「谁是卧底」引擎，带 **LangGraph checkpoint + `interrupt` / `Command` 人机恢复**，以及 **Streamlit** 本机 Web 界面。

## 玩法说明

- **人数**：4～10 人；界面默认常用规模为 **5 人**（可自行改为 4～6 人 + **1 名卧底**）。
- **卧底数量**：
  - 人数 **≤ 6**：固定 **1** 名卧底（与教材一致）。
  - 人数 **> 6**：可选择 **1 或 2** 名卧底（双卧底互不知身份，词均为卧底词）。
- **角色**：仅 **平民 / 卧底**，**不包含白板**。
- **流程**：系统出题 → 随机分配角色与词语 → 存活玩家依次发言 → 投票 → **得票最高者淘汰**；若平票，在得票最高者中 **随机淘汰一名**（与教材一致）。
- **胜负**：
  - **平民胜**：任意时刻 **所有卧底均已出局**。
  - **卧底胜**：场上剩余 **恰好 1 名平民且仍有至少 1 名卧底**。
- **最大轮次**：超过配置的上限仍未结束时，判定 **平民方胜利（超时）**，防止死循环。
- **模式**：
  - **观战**：人类玩家选「无」，所有发言与投票由 LLM 完成。
  - **入局**：在侧栏选择 `P1`…`Pn` 之一作为人类；轮到该玩家时，Web 会弹出发言/投票表单，通过 **`Command(resume=...)`** 从 checkpoint 恢复执行。

## 环境变量（标准命名）

复制 `.env.example` 为 `.env`按需填写：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | API Key |
| `OPENAI_API_BASE` | 兼容 OpenAI 的 Base URL（魔搭 ModelScope 等；可留空走默认） |
| `OPENAI_MODEL` | 模型名 |
| `OPENAI_TEMPERATURE` | 可选 |
| `OPENAI_MAX_TOKENS` | 可选 |

## Python 与依赖

- **Python**：3.10+
- **安装**：

```bash
cd WhoIsTheSpy
pip install -r requirements.txt
```

## 运行方式

### 1）Streamlit（推荐，含配置、可视化、人机）

默认在本机启动，浏览器访问终端中提示的地址，一般为：

**http://localhost:8501**

```bash
cd WhoIsTheSpy
streamlit run app.py
```

侧栏可配置：大模型参数、人数与卧底数、人类座位、最大轮次。主界面包含：

- **对局**：观战或人类提交发言/投票（checkpoint 恢复）。
- **LangGraph 可视化**：将拓扑导出为 **PNG**（`get_graph().draw_mermaid_png()`，默认写入 `.cache/langgraph_topology.png` 并在页面加载）+ 本局 **节点执行顺序**（`trace`）。
- **日志与状态**：文本日志与 checkpoint `values` 快照。

### 2）控制台（全 AI 一局）

```bash
cd WhoIsTheSpy
python who_is_undercover.py
```

## LangGraph 结构说明

节点大致为：

`generate_words` → `assign_roles` → `begin_speech_round` →（循环）`speech_turn` → `finalize_speeches` → `vote_begin` →（循环）`vote_turn` → `judge_result` → 条件回到 `begin_speech_round` 或 `show_final_result`。

- **checkpoint**：`MemorySaver`，按 `thread_id` 区分对局。
- **人机**：在 `speech_turn` / `vote_turn` 中对人类玩家调用 `interrupt(...)`；前端用 `Command(resume=...)` 恢复。

## 与课程章节对应关系

- **状态与图**：第六章 / 第七章 — `TypedDict`、`StateGraph`、`START` / `END`、条件边、循环。
- **模型与链**：第二章 — `ChatPromptTemplate`、LCEL `prompt | llm | parser`。
- **环境变量**：第一章 — `python-dotenv` 与 `ChatOpenAI` 兼容网关。

## 目录结构

```
WhoIsTheSpy/
├── app.py                 # Streamlit 入口
├── who_is_undercover.py   # 控制台全 AI 一局
├── readme.md
├── requirements.txt
├── .env.example
├── .cache/                # 运行后：langgraph_topology.png
└── spy_game/
    ├── config.py
    ├── state_types.py
    ├── llm_factory.py
    ├── utils.py
    ├── engine_nodes.py
    ├── graph_build.py
    └── runner.py
```

## 文件功能说明

| 文件 | 功能简述 |
|------|----------|
| `readme.md` | 项目说明：玩法、环境变量、安装与运行、LangGraph 结构、目录与**本表**（各文件职责）。 |
| `requirements.txt` | 列出运行所需的 Python 包及版本下限，供 `pip install -r` 安装。 |
| `.env.example` | 环境变量**键名**示例（不含真实密钥）；复制为 `.env` 后填写。 |
| `.gitignore` | 指定 Git 不跟踪的路径（如 `.venv/`、`.cache/`、`.env`）。 |
| `app.py` | **Streamlit** 入口：侧栏大模型与对局配置、「开始新对局」、拓扑 **PNG**、运行日志与 checkpoint 快照、人机发言/投票与 `resume`。 |
| `who_is_undercover.py` | **命令行**入口：全 AI 观战一局，控制台打印流程与最终状态。 |
| `spy_game/__init__.py` | 包初始化，标识 `spy_game` 为 Python 包。 |
| `spy_game/config.py` | 配置数据类：`LLMConfig`（模型参数）、`GameConfig`（人数/卧底/人类位等）、`RuntimeContext`（运行期 LLM、对局配置、日志与 trace）。 |
| `spy_game/state_types.py` | `GameState`（TypedDict）字段定义与 `empty_state()` 初始状态工厂。 |
| `spy_game/llm_factory.py` | 读取 `.env`（及上级目录 `.env`），构造 `ChatOpenAI` 与 `StrOutputParser`，支持侧栏覆盖参数。 |
| `spy_game/utils.py` | 通用小工具（如剥离 LLM 输出外的 Markdown 代码围栏）。 |
| `spy_game/engine_nodes.py` | **LangGraph 节点实现**：出题、分角、每轮发言/投票单步、归档发言、裁决、结算；含 `interrupt` 人机点与路由函数。 |
| `spy_game/graph_build.py` | **图定义**：`StateGraph` 注册节点与边、条件边；`compile_game_graph`；`get_graph_mermaid` / `export_graph_topology_png` 导出拓扑。 |
| `spy_game/runner.py` | **运行封装**：`MemorySaver` 编译、`start_new_game`、`resume_game`（`Command(resume=…)`）、`get_snapshot`、`extract_interrupt_payload` 等。 |
| `.cache/langgraph_topology.png` | **运行后自动生成**（非手写源码）：由 `draw_mermaid_png` 写入，供界面展示 LangGraph 拓扑。 |
