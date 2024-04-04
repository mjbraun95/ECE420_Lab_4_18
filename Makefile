all:
	mpicc main.c Lab4_IO.c -o main -lm

clean:
	rm -f main data_output data_input_link data_input_meta 