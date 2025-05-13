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