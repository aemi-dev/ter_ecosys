.PHONY: clean run

run: ./ecco/rr/__init__.py
	python3 ./mkecco.py
	python3 ./ecco/__init__.py
	ipython3

clean:
	rm -rf *.pnml


# o.write(open('punf.dot','w'),fmt="dot",m=0) 
# dot -T pdf punf.dot -o punf.dot.pdf