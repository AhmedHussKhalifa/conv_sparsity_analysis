#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <math.h>


using  namespace std;

int main(){
	string a[4995];
	int h[4995],w[4995];
	ifstream file_list("../dataset/vgg19/file_list");
	for(int i=0;i<4995;i++)
		file_list>>a[i];
	file_list.close();
	ifstream conv_shape("../dataset/vgg19/conv_shape");
	for(int i=0;i<4995;i++){
		conv_shape>>h[i];
		w[i] = h[i];
	}
	conv_shape.close();
	double len,temp;
	
	for(int i=0;i<4995;i++){
		int i_w = w[i];
		int i_h = h[i];
		double arraySize = i_w*i_h;
		
		float *feature = new float[int(arraySize)];
		
		len = 0;
		ifstream conv_feature(("../dataset/vgg19/conv/"+a[i]).c_str());
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
		cout<<sparse<<endl;
		free(feature);
	}
	
	return 0;	
} 
	
	
