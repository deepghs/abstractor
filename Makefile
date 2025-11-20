.PHONY: ds_clone ds_pull ds_save ds_push

PYTHON    := $(shell which python)
PAGES_DIR ?= pages

ds_clone:
	git clone "https://${GH_ACCESS_TOKEN}@github.com/deepghs/deepghs.github.io.git" "${PAGES_DIR}"
	cd "${PAGES_DIR}" && git config user.name narugo1992 && git config user.email narugo1992@deepghs.org

ds_pull:
	cd "${PAGES_DIR}" && git pull

ds_save:
	cd "${PAGES_DIR}" && \
	if [ -n "$$(git status --porcelain)" ]; then \
		echo "Changes detected, staging files..."; \
		git add -A; \
		git commit -m "Auto update: $$(date)"; \
		echo "Files staged successfully"; \
	else \
		echo "No changes detected, nothing to stage"; \
	fi

ds_push:
	cd "${PAGES_DIR}" && git push
