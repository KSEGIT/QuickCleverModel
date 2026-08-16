# Thin wrapper over stack.sh so `make up` works alongside `./stack.sh up`.
.PHONY: help up down restart status logs open bench reap cache-viz models menubar

APP := build/BonsaiMenuBar.app

help:           ## Show this help
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n",$$1,$$2}'

models:         ## Download the GGUF weights (~11 GB, both models — not part of `make up`)
	@./fetch-models.sh

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

cache-viz:      ## Live prompt-cache dashboard on :8090
	@pkill -f cache-viz.py 2>/dev/null || true
	@mkdir -p run/logs && nohup python3 cache-viz.py > run/logs/cacheviz.log 2>&1 & sleep 2; \
	  echo "  cache visualiser -> http://127.0.0.1:8090"; \
	  (open http://127.0.0.1:8090 2>/dev/null || xdg-open http://127.0.0.1:8090 >/dev/null 2>&1 &)

open:           ## Open the chat UI
	@./stack.sh open

bench:          ## Median throughput over N reps — make bench MODEL=bonsai-27b-1bit REPS=9
	@./bench.sh -n "$(or $(REPS),5)" -m "$(or $(MODEL),bonsai-27b-ternary)"

menubar:        ## Build the menu bar app into build/ (needs Xcode)
	@mkdir -p $(APP)/Contents/MacOS
	@sed -e "s|@ROOT@|$(CURDIR)|g" menubar/Info.plist.in > $(APP)/Contents/Info.plist
	@swiftc -O -swift-version 5 -parse-as-library \
	  -o $(APP)/Contents/MacOS/BonsaiMenuBar menubar/BonsaiMenuBar.swift
	@echo "  built $(APP)"
