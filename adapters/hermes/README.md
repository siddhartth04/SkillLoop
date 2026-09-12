# Hermes Agent adapter

Two pieces:

1. **MCP server** — add SkillLoop to Hermes' MCP config (check the current Hermes docs for the exact key; it is
   `mcp_servers` in `~/.hermes/config.yaml` at time of writing):

   ```yaml
   mcp_servers:
     skillloop:
       command: skillloop
       args: ["mcp"]
       env:
         ANTHROPIC_API_KEY: "..."   # or OPENAI_API_KEY + OPENAI_BASE_URL
   ```

2. **Skill** — copy `skillloop-learn/` into `~/.hermes/skills/`. Its description is deliberately pushy so
   Hermes calls recall/learn around real tasks.

Optionally point Hermes' skill directory at `skillloop path` so gated skills are also visible natively.
