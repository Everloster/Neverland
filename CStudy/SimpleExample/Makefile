GetOptTest : GetOptTest.o libGetEnvTest
	gcc $(CFLAGS) -o GetOptTest GetOptTest.o -L. -lGetEnvTest

GetEnvTest.o : GetEnvTest.c
	gcc $(CFLAGS) -c GetEnvTest.c

GetOptTest.o : GetOptTest.c
	gcc $(CFLAGS) -c GetOptTest.c

libGetEnvTest : GetEnvTest.o
	gcc $(CFLAGS) -shared -fPIC -o libGetEnvTest.so GetEnvTest.o

LoadSoTest : libGetEnvTest LoadSoTest.o
	gcc $(CFLAGS) -o LoadSoTest LoadSoTest.c

clean : 
	rm -rf *.o GetOptTest GetEnvTest
	rm -rf libGetEnvTest.so
	rm -rf LoadSoTest
