1. mcp 和 tool call 的区别
2. Why not use a single monolithic agent?
    Separating order logic from FAQ avoids polluting the retrieval context with
    structured API data, and keeps each agent's prompt shorter and more focused.