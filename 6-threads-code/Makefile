all: msg

msg: semaphore.o msg_sem.o
	clang++ -std=c++11 -stdlib=libc++ semaphore.o msg_sem.o -o msg

semaphore.o: semaphore.h
	clang++ -c -std=c++11 -stdlib=libc++ semaphore.h -o semaphore.o

msg_sem.o: th_msg_sem.cpp semaphore.o
	clang++ -c -std=c++11 -stdlib=libc++ th_msg_sem.cpp 

