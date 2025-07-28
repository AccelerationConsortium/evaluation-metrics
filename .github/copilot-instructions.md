## Development Practices

- Start with minimal, lean implementations focused on proof-of-concept
- Avoid implementing things from scratch
- Avoid defensive error handling for hypothetical failures
- Use print statements and logging sparingly, unless asked
- Avoid light wrappers and custom classes, unless asked
- Avoid `if __name__ == "__main__"` patterns in package code
- Skip unit tests unless explicitly requested
- Follow patterns in CONTRIBUTING.md when present

## Git Operations

- When asked to remove all file changes, use `git checkout -- <filename>`
- Copy-paste commands you run and summarized execution status directly in your comment replies

## External Resources

- Validate and access link content using available MCP tools (Playwright and/or Perplexity)
- Search GitHub for relevant open-source alternatives to commercial tools
- Prioritize official package documentation over inferred usage

## Communication Style

- Use minimal emoji and special symbols
- Prioritize clarity and brevity in responses
- Ask clarifying questions when needed
- Don't infer requirements or create workarounds unless asked
- Put documentation content in comment replies, not separate files, unless asked

## Repo-specific

- Activate the `balam-env` conda environment before running code (even if running on Niagara) or use a venv
- When asked to SSH into BALAM, use `sgbaird@balam.scinet.utoronto.ca` and the `CCDB_SSH_PRIVATE_KEY`
- When asked to SSH into Niagara, use `sgbaird@niagara.scinet.utoronto.ca` and the same `CCDB_SSH_PRIVATE_KEY`
- Running test scripts locally is fine, but the only success metric is submitting jobs to the cluster via submitit and having those jobs complete successfully on the corresponding compute nodes with the right dependencies available. Nothing else counts
- Use the perplexity MCP tool to refer to the BALAM (https://docs.scinet.utoronto.ca/index.php/Balam) and Niagara (https://docs.scinet.utoronto.ca/index.php/Niagara_Quickstart) docs to ensure consistency with cluster-specific requirements
- After SSH'ing into the cluster, make sure to type "1" and "press enter" to trigger the Duo authentication step, which I'll approve manually on my end
- Don't create any command line interfaces unless explicitly requested
- If the SSH bash output gets too long and unwieldy, you can log out and recreate the SSH connection to refresh the terminal text

<!--- add as .github/copilot-instructions.md, see https://docs.github.com/en/enterprise-cloud@latest/copilot/using-github-copilot/coding-agent/best-practices-for-using-copilot-to-work-on-tasks for additional context --->
