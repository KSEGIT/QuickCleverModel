# Thin wrapper over stack.sh so `make up` works alongside `./stack.sh up`.
.PHONY: help up down restart status logs open bench reap

help:           ## Show this help
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n",$$1,$$2}'

up:             ## Start the whole stack (llama + playwright + webui)
	@./stack.sh up

down:           ## Stop everything
	@./stack.sh down

restart:        ## Restart everything
	@./stack.sh restart

status:         ## Show what is running
	@./stack.sh status

logs:           ## Tail all logs (make logs SVC=llama for one)
	@./stack.sh logs $(SVC)

reap:           ## Kill stale/orphaned service processes after a crash
	@./stack.sh reap

open:           ## Open the chat UI
	@./stack.sh open

bench:          ## Measure generation throughput against the running server
	@set -a; . ./.env; set +a; \
	curl -s --max-time 300 http://127.0.0.1:8080/v1/chat/completions \
	  -H "Content-Type: application/json" -H "Authorization: Bearer $$BONSAI_API_KEY" \
	  -d '{"messages":[{"role":"user","content":"Count to twenty."}],"max_tokens":150}' \
	| python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('  %.1f tok/s generation, %.1f t/s prompt'%(t['predicted_per_second'],t['prompt_per_second']))"
