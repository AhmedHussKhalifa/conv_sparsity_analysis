#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <math.h>
#include <sys/types.h>
#include <dirent.h>

// #define fileNum 190 
using  namespace std;

int count_dir()
{
	DIR *dp;
	int i = 0;
	struct dirent *ep;     
	dp = opendir ("../conv/");

	if (dp != NULL)
	{
		while (ep = readdir (dp))
		i++;
		(void) closedir (dp);
	}
	else
	{
		perror ("Couldn't open the directory");
	}
	printf("There's %d files in the current directory.\n", i);
	return i;
}

int main(){

	// int fileNum = count_dir();
	int fileNum = 3 * 94;
	string a[fileNum];
	int h[fileNum],w[fileNum];
	ifstream file_list("../gen/file_list");
	for(int i=0;i<fileNum;i++)
		file_list>>a[i];
	file_list.close();
	ifstream conv_shape("../gen/conv_shape");
	for(int i=0;i<fileNum;i++){
		conv_shape>>h[i];
		w[i] = h[i];
	}
	conv_shape.close();
	double len,temp;
	
	for(int i=0;i<fileNum;i++){
		int i_w = w[i];
		int i_h = h[i];
		double arraySize = i_w*i_h;
		
		float *feature = new float[int(arraySize)];
		
		len = 0;
		ifstream conv_feature(("../conv/"+a[i]).c_str());
		while(!conv_feature.eof())
			conv_feature>>feature[int(len++)];
		conv_feature.close();
		
		temp=0;
		for(int j=0;j<arraySize;j++){
			//cout<<feature[j]<<endl;
			if(feature[j]==0){
				temp ++;
				//cout<<feature[j]<<endl;
			}
		}
		double sparse = double(temp/arraySize);
		// cout<< i << "--> "<<sparse<<endl;
		cout<<sparse<<endl;
		free(feature);
	}
	
	return 0;	
} 
	
	
