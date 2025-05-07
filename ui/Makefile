CHECKDIRS := src


# Run checks on all files for the repo
quality:
	@echo "Running quality checks";
	@# Run eslint checks
	npx eslint --max-warnings 10 $(CHECKDIRS)
	@# Run TypeScript checks
	npx tsc --noEmit

# Style the code according to accepted standards for the repo
style:
	@echo "Running styling";
	@# Auto fix any eslint issues
	npx eslint --fix $(CHECKDIRS)

test-unit:
	@echo "Running unit tests";
	@# Run react tests
	npm run test:unit

test-integration:
	@echo "Running integration tests";
	@# Run react tests
	npm run test:integration