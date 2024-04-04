all: template
	mpicc main.c Lab4_IO.c -o main -lm

template:
	gcc main_template.c Lab4_IO.c -o main_template -lm

clean:
	rm -f main main_template data_output data_input_link data_input_meta