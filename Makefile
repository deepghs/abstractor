.PHONY: docs test unittest resource antlr antlr_build build package clean

PYTHON    := $(shell which python)
PAGES_DIR ?= pages

ds_clone:
	git clone "https://${GH_ACCESS_TOKEN}@github.com/deepghs/deepghs.github.io.git" "${PAGES_DIR}"
	cd "${PAGES_DIR}" && git config user.name narugo1992 && git config user.email narugo1992@deepghs.org
ds_pull:
	cd "${PAGES_DIR}" && git pull

