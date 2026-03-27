# Makefile for building documentation

.PHONY: docs html clean

docs:
	sphinx-apidoc -f -o source/api ../go2_mpc
	sphinx-build -b html source build/html

html:
	sphinx-build -b html source build/html

clean:
	rm -rf build/html build/doctrees
	rm -rf source/api/*.rst

serve:
	cd build/html && python -m http.server 8000
