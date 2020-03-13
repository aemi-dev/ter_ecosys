.PHONY: clean run

run: ./ecco/rr/__init__.py
	python3 ./mkecco.py
	python3 ./ecco/__init__.py
	ipython3 --no-banner -c run -m ecco models/termites-simpler.rr "model.unfold(unf='punf')"

clean:
	rm cuf.bak.pnml
