ROOT := $(shell pwd) 

.PHONY: tidy
tidy: 
	@echo "Tidying up the go.mod and go.sum files"
	@go fmt .
	@go mod tidy

.PHONY: run
run: 
	@go run main.go

