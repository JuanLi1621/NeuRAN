#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>

std::string inPath = "./data/";
std::string outPath = "./data/";
std::string negPath = "";

extern "C"
void setInPath(char *path) {
	INT len = strlen(path);
	inPath = "";
	for (INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input Files Path : %s\n", inPath.c_str());
}

extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}

extern "C"
void setNegPath(char *path) {
	INT len = strlen(path);
	negPath = "";
	for (INT i = 0; i < len; i++)
		negPath = negPath + path[i];
	printf("Noise Files Path : %s\n", negPath.c_str());
}

/*
============================================================
*/

INT workThreads = 1;

extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

extern "C"
INT getWorkThreads() {
	return workThreads;
}

/*
============================================================
*/

INT relationTotal = 0;
INT entityTotal = 0;//
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;//
INT trainPosTotal = 0;
INT trainNoiTotal = 0;
INT testTotal_neg = 0;
INT validTotal_neg = 0;

INT hd_nbrs_Total = 0;
INT tl_nbrs_Total = 0;
INT hd_max = 0;
INT tl_max = 0;

extern "C"
INT getHd_nbrs_Total() {
	return hd_nbrs_Total;
}

extern "C"
INT getTl_nbrs_Total() {
	return tl_nbrs_Total;
}

extern "C"
INT getHd_max(){
    return hd_max;
}

extern "C"
INT getTl_max(){
    return tl_max;
}

extern "C"
INT getEntityTotal() {
	return entityTotal;
}

extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return tripleTotal;
}

extern "C"
INT getTrainTotal() {
	return trainTotal;
}

extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}

extern "C"
INT getTrainPosTotal() {
	return trainPosTotal;
}

extern "C"
INT getTrainNoiTotal() {
	return trainNoiTotal;
}

/*
============================================================
*/

INT bernFlag = 0;

extern "C"
void setBern(INT con) {
	bernFlag = con;
}

#endif
