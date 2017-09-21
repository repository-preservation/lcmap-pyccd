profile:
	kernprof -v -l pytest

build_cython:
	python setup.py build_ext --inplace

clean_cython:
	rm ccd/*.c
	rm ccd/*.so
	rm ccd/models/*.c
	rm ccd/models/*.so
