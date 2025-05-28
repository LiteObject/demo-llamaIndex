# Multi-Agent Workflow Example Overview

This example demonstrates a multi-agent workflow using LlamaIndex, where several specialized agents collaborate to research, write, and review a comprehensive report on a given topic. The workflow is designed to automate the process of gathering information, synthesizing it into a report, and ensuring quality through review.

## Agents in the Workflow

### 1. ResearchAgent
- **Purpose:** Searches the web for information on a given topic and records research notes.
- **Tools:**
  - Web search tool (e.g., Tavily)
  - `record_notes` function to store findings in the agent's state
- **Workflow:**
  - Gathers up-to-date information from trusted sources.
  - Records notes under specific titles for later use.
  - Once research is complete, hands off control to the WriteAgent.

### 2. WriteAgent
- **Purpose:** Writes a detailed report based on the research notes.
- **Tools:**
  - `write_report` function, which saves the report content to the agent's state and writes it as a Markdown file in the `docs/` folder (with a timestamped filename to avoid duplicates).
- **Workflow:**
  - Synthesizes the research notes into a well-structured markdown report.
  - Ensures the report is grounded in the collected research.
  - After writing, requests feedback from the ReviewAgent.

### 3. ReviewAgent
- **Purpose:** Reviews the written report and provides feedback.
- **Tools:**
  - `review_report` function to record feedback in the agent's state.
- **Workflow:**
  - Reads the generated report.
  - Provides feedback, either approving the report or requesting changes.
  - Can hand control back to the WriteAgent for revisions if needed.

## How the Workflow Operates
1. **Initialization:**
   - The workflow is started with a user prompt specifying the topic and requirements (e.g., a 5,000-word blog post on recent health science developments).
   - The initial state includes empty research notes, a placeholder for the report, and a review status.
2. **Research Phase:**
   - The ResearchAgent uses web search tools to gather information and records notes.
   - Once satisfied, it hands off to the WriteAgent.
3. **Writing Phase:**
   - The WriteAgent composes a markdown report using the research notes.
   - The report is saved in the `docs/` directory with a unique timestamped filename.
   - The agent then requests a review.
4. **Review Phase:**
   - The ReviewAgent reviews the report and provides feedback.
   - If changes are needed, the WriteAgent can revise the report; otherwise, the process concludes.

## Key Features
- **Separation of Concerns:** Each agent has a clear, specialized role.
- **State Management:** Agents share and update a common state, passing information and results between phases.
- **File Output:** Reports are automatically saved as markdown files in the `docs/` folder, with filenames that include a timestamp to prevent overwriting.
- **Extensibility:** The workflow can be extended with additional agents or tools as needed.

## Usage
- Run the script directly. The agents will collaborate to fulfill the user prompt, and the final report will be saved in the `docs/` directory.
- The process and agent actions are streamed to the console for transparency and debugging.

---

This example provides a foundation for building more complex, automated multi-agent systems for research, writing, and review tasks.
