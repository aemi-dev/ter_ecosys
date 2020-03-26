.PHONY: clean run

run: ./ecco/rr/__init__.py
	python3 ./mkecco.py
	python3 ./ecco/__init__.py
	ipython3

clean:
	rm -rf *.pnml
