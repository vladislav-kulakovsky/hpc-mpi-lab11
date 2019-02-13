#include <mpi.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iterator>

#define N 10

// std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
std::vector<int> v;

void p2p();
void collective_share();

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	
	std::srand(time(0));
	for(int i = 0; i < N; ++i)
		v.push_back(std::rand() % 10 + 1);

	p2p();
	MPI_Barrier(MPI_COMM_WORLD);

	collective_share();

	MPI_Finalize();
	return 0;
}

void p2p()
{
	int rank = 0, proccessNum = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proccessNum);

	if (rank == 0)
	{
		unsigned long int result = 0;
		int len = v.size();
		for (int i = 0; i < len; ++i)
		{
			if (i == len -1)
			{
				MPI_Status stat;
				MPI_Send(&i, sizeof(int), MPI_INT, (i % (proccessNum - 1)) + 1, 1, MPI_COMM_WORLD);
				MPI_Recv(&result, sizeof(int), MPI_UNSIGNED_LONG, (i % (proccessNum - 1)) + 1, 1, MPI_COMM_WORLD, &stat);
				printf("P2P Result = %lld\n", result);
			}
			else
			{
				MPI_Send(&i, sizeof(int), MPI_INT, (i % (proccessNum - 1)) + 1, 1, MPI_COMM_WORLD);
			}
		}

		MPI_Comm_size(MPI_COMM_WORLD, &proccessNum);

		for(int i = 1; i < proccessNum; ++i)
		{
			int exit = -1;
			MPI_Send(&exit, sizeof(int), MPI_INT, i, 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		int iter = 0;
		while(iter != -1)
		{
			MPI_Status stat;
			unsigned long int res = 1LL;
			MPI_Recv(&iter, sizeof(int), MPI_INT, 0, 1, MPI_COMM_WORLD, &stat);
			if (iter == -1)
				continue;
			for (int i = 0; i <= iter; ++i)
			{
				res *= v[i];
			}
			if (iter == v.size() - 1)
			{
				MPI_Send(&res, sizeof(int), MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD);
			}

			printf("process #%d\tC[1..%d] = %lld\n", rank, iter, res);
		}
	}
}

void collective_share()
{
	int rank = 0, proccessNum = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proccessNum);

	int need_size = v.size();
	if (v.size() % proccessNum != 0)
		need_size = v.size() + (proccessNum - (v.size() % proccessNum));

	int sendcount = need_size / proccessNum;
	int recvcount = sendcount;
	int *recvbuf = new int[recvcount];
	int *sendbuf = new int[need_size];
	long long int *gathered = new long long int[need_size];

	if (rank == 0)
	{
		for (int i = 0; i < v.size(); ++i)
			sendbuf[i] = i;
		for (int i = v.size(); i < need_size; ++i)
		{
			sendbuf[i] = -1;
		}
	}
	MPI_Scatter(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	long long int *result = new long long int[recvcount];

	for(int i = 0; i < recvcount; ++i)
	{
		result[i] = 1;
		if(recvbuf[i] != -1)
		{
			for(int j = 0; j <= recvbuf[i]; ++j)
			{
				result[i] *= v[j];
			}
			printf("s_process #%d\tC[1..%d] = %lld\n", rank, recvbuf[i], result[i]);
		}
	}

	MPI_Gather(result, recvcount, MPI_LONG_LONG_INT, gathered, sendcount, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
	{
		std::vector<long long int> result_vector;
		result_vector.insert(result_vector.end(), gathered, gathered + need_size);
		printf("Share Result = %lld\n", *std::max_element(result_vector.begin(), result_vector.end()));
	}
}
