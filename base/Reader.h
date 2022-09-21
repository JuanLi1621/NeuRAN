#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include "Domain.h"
#include "Range.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <string>
using namespace std;

INT *freqRel, *freqEnt; 
INT *lefHead, *rigHead; 
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
REAL *left_mean, *right_mean;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

Domain *trainHdNbrs; //rels of each head entity
Range *trainTlNbrs; //rels for each tail entity
INT *hd_nbrs_lef, *hd_nbrs_rig, *tl_nbrs_lef, *tl_nbrs_rig;

extern "C"
void importTrainFiles() {
	printf("The toolkit is importing training set.\n");
	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	if (fin == nullptr) {
		std::cout << '`' << inPath << "relation2id.txt" << '`' << " does not exist"
		          << std::endl;
		return;
	}
	tmp = fscanf(fin, "%ld", &relationTotal); 
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	if (fin == nullptr) {
		std::cout << '`' << inPath << "entity2id.txt" << '`' << " does not exist"
		          << std::endl;
		return;
	}
	tmp = fscanf(fin, "%ld", &entityTotal); 
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	if (fin == nullptr) {
		std::cout << '`' << inPath << "train2id.txt" << '`' << " does not exist"
		          << std::endl;
		return;
	}
	tmp = fscanf(fin, "%ld", &trainTotal);  
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));    
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));  

	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);    
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head); //(h,r,t)

	tmp = trainTotal;
	trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1; 
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++){
		if (trainList[i].h != trainList[i - 1].h ||
		    trainList[i].r != trainList[i - 1].r ||
		    trainList[i].t != trainList[i - 1].t) {
	        trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
	        freqEnt[trainList[i].t]++;
	        freqEnt[trainList[i].h]++;
	        freqRel[trainList[i].r]++;
        } else {
			printf("the same triples: i=%ld\n", i);
		}
    }   
	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head); //cmp_head:sort by (h, r, t)
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail); //cmp_tail:sort by (t, r, h)
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel); //cmp_rel:sort by (h, t, r)
	printf("The total of train triples is %ld.\n", trainTotal);

	//#	dm_nbrs
	fin = fopen((inPath + "ori_att_h_out_ranked.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &hd_nbrs_Total);
	tmp = fscanf(fin, "%ld", &hd_max);
	trainHdNbrs = (Domain *)calloc(hd_nbrs_Total, sizeof(Domain));
	hd_nbrs_Total = 0;
	while (fscanf(fin, "%ld", &trainHdNbrs[hd_nbrs_Total].h) == 1){
		tmp = fscanf(fin, "%ld", &trainHdNbrs[hd_nbrs_Total].r);
		tmp = fscanf(fin, "%f", &trainHdNbrs[hd_nbrs_Total].pro);
		hd_nbrs_Total++;
	}
    fclose(fin);
    std::sort(trainHdNbrs, trainHdNbrs + hd_nbrs_Total, Domain::cmp_hr); //sort by (h,r,pro)
    printf("The total of hd_nbrs is %ld.\n", hd_nbrs_Total);

	//# rg_nbrs
	fin = fopen((inPath + "ori_att_t_in_ranked.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &tl_nbrs_Total);
	tmp = fscanf(fin, "%ld", &tl_max);
	trainTlNbrs = (Range *)calloc(tl_nbrs_Total, sizeof(Range));
	tl_nbrs_Total = 0;
	while (fscanf(fin, "%ld", &trainTlNbrs[tl_nbrs_Total].t) == 1){
		tmp = fscanf(fin, "%ld", &trainTlNbrs[tl_nbrs_Total].r);
		tmp = fscanf(fin, "%f", &trainTlNbrs[tl_nbrs_Total].pro);
		tl_nbrs_Total++;
	}
	fclose(fin);
    std::sort(trainTlNbrs, trainTlNbrs + tl_nbrs_Total, Range::cmp_tr); //sort by (t,r,pro)
    printf("The total of tl_nbrs is %ld.\n", tl_nbrs_Total);

    hd_nbrs_lef = (INT *)calloc(entityTotal, sizeof(INT));
    hd_nbrs_rig = (INT *)calloc(entityTotal, sizeof(INT));
    tl_nbrs_lef = (INT *)calloc(entityTotal, sizeof(INT));
    tl_nbrs_rig = (INT *)calloc(entityTotal, sizeof(INT));
    memset(hd_nbrs_rig, -1, sizeof(INT) * entityTotal);
	memset(tl_nbrs_rig, -1, sizeof(INT) * entityTotal);

	for (INT i = 1; i < hd_nbrs_Total; i++) {
		if (trainHdNbrs[i].h != trainHdNbrs[i - 1].h) {
			hd_nbrs_rig[trainHdNbrs[i - 1].h] = i - 1;
			hd_nbrs_lef[trainHdNbrs[i].h] = i;
		}
	}
	hd_nbrs_lef[trainHdNbrs[0].h] = 0;
	hd_nbrs_rig[trainHdNbrs[hd_nbrs_Total - 1].h] = hd_nbrs_Total - 1;
	
	for (INT i = 1; i < tl_nbrs_Total; i++) {
	    if (trainTlNbrs[i].t != trainTlNbrs[i - 1].t) {
			tl_nbrs_rig[trainTlNbrs[i - 1].t] = i - 1;
			tl_nbrs_lef[trainTlNbrs[i].t] = i;
		}
	}
	tl_nbrs_lef[trainTlNbrs[0].t] = 0;
	tl_nbrs_rig[trainTlNbrs[tl_nbrs_Total - 1].t] = tl_nbrs_Total - 1;

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));

	memset(rigHead, -1, sizeof(INT) * entityTotal);
	memset(rigTail, -1, sizeof(INT) * entityTotal);
	memset(rigRel, -1, sizeof(INT) * entityTotal);

	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	left_mean = (REAL *)calloc(relationTotal, sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal, sizeof(REAL));

	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
	printf("finish import train files.\n");
}

Triple *testList, *testList_neg;
Triple *validList, *validList_neg;
Triple *tripleList;
INT *testLef, *testRig;
INT *validLef, *validRig;

Triple *trainPosList, *trainNoiList;
INT *noiPosLef, *noiPosRig;
INT *noiNegLef, *noiNegRig;

extern "C"
void importTestFiles() {
	FILE *fin;
	INT tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	if (fin == nullptr) {
		std::cout << '`' << inPath << "relation2id.txt" << '`' << " does not exist"
		          << std::endl;
		return;
	}
	tmp = fscanf(fin, "%ld", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	if (fin == nullptr) {
		std::cout << '`' << inPath << "entity2id.txt" << '`' << " does not exist"
		          << std::endl;
		return;
	}
	tmp = fscanf(fin, "%ld", &entityTotal);
	fclose(fin);

	//noise detection
    fin = fopen((inPath + "train2id_origin.txt").c_str(), "r");//training data without generated noisy triples
	tmp = fscanf(fin, "%ld", &trainPosTotal);  //trainTotal
	trainPosList = (Triple *)calloc(trainPosTotal, sizeof(Triple));
	for (INT i = 0; i < trainPosTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainPosList[i].h);
		tmp = fscanf(fin, "%ld", &trainPosList[i].t);
		tmp = fscanf(fin, "%ld", &trainPosList[i].r);
	}
	fclose(fin);    //trainPosList

	fin = fopen((inPath + negPath).c_str(), "r"); //neg_path
	tmp = fscanf(fin, "%ld", &trainNoiTotal);  //noisy triples with different ratios
	trainNoiList = (Triple *)calloc(trainNoiTotal, sizeof(Triple));
	for (INT i = 0; i < trainNoiTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainNoiList[i].h);
		tmp = fscanf(fin, "%ld", &trainNoiList[i].t);
		tmp = fscanf(fin, "%ld", &trainNoiList[i].r);
	}
	fclose(fin);    //trainNoiList
	
	std::sort(trainPosList, trainPosList + trainPosTotal, Triple::cmp_rel2); //sorted by (r,h,t)
	noiPosLef = (INT *)calloc(relationTotal, sizeof(INT));
	noiPosRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(noiPosLef, -1, sizeof(INT) * relationTotal);
	memset(noiPosRig, -1, sizeof(INT) * relationTotal);
	for (INT i = 1; i < trainPosTotal; i++) {
		if (trainPosList[i].r != trainPosList[i - 1].r) {
			noiPosRig[trainPosList[i - 1].r] = i - 1;
			noiPosLef[trainPosList[i].r] = i;
		}
	}
	noiPosLef[trainPosList[0].r] = 0;
	noiPosRig[trainPosList[trainPosTotal - 1].r] = trainPosTotal - 1;

	std::sort(trainNoiList, trainNoiList + trainNoiTotal, Triple::cmp_rel2);
	noiNegLef = (INT *)calloc(relationTotal, sizeof(INT));
	noiNegRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(noiNegLef, -1, sizeof(INT) * relationTotal);
	memset(noiNegRig, -1, sizeof(INT) * relationTotal);
	for (INT i = 1; i < trainNoiTotal; i++) {
		if (trainNoiList[i].r != trainNoiList[i - 1].r) {
			noiNegRig[trainNoiList[i - 1].r] = i - 1;
			noiNegLef[trainNoiList[i].r] = i;
		}
	}
	noiNegLef[trainNoiList[0].r] = 0;
	noiNegRig[trainNoiList[trainNoiTotal - 1].r] = trainNoiTotal - 1;

	FILE *f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
	FILE *f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
	FILE *f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
	FILE *f_kb4 = fopen((inPath + "valid2id_neg.txt").c_str(), "r");
	FILE *f_kb5 = fopen((inPath + "test2id_neg.txt").c_str(), "r");
	tmp = fscanf(f_kb1, "%ld", &testTotal);
	tmp = fscanf(f_kb2, "%ld", &trainTotal);
	tmp = fscanf(f_kb3, "%ld", &validTotal);
	tmp = fscanf(f_kb4, "%ld", &validTotal_neg);
	tmp = fscanf(f_kb5, "%ld", &testTotal_neg);
	tripleTotal = testTotal + trainTotal + validTotal;
	testList = (Triple *)calloc(testTotal, sizeof(Triple));
	validList = (Triple *)calloc(validTotal, sizeof(Triple));
	tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	for (INT i = 0; i < testTotal; i++) {
		tmp = fscanf(f_kb1, "%ld", &testList[i].h);
		tmp = fscanf(f_kb1, "%ld", &testList[i].t);
		tmp = fscanf(f_kb1, "%ld", &testList[i].r);
		tripleList[i] = testList[i];
	}
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
		tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
	}
	for (INT i = 0; i < validTotal; i++) {
		tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
		tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
		tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
		validList[i] = tripleList[i + testTotal + trainTotal];
	}
	//neg triples for triple classification
	testList_neg = (Triple *)calloc(testTotal, sizeof(Triple));
	validList_neg = (Triple *)calloc(validTotal, sizeof(Triple));
	for (INT i = 0; i < validTotal_neg; i++) {
		tmp = fscanf(f_kb4, "%ld", &validList_neg[i].h);
		tmp = fscanf(f_kb4, "%ld", &validList_neg[i].t);
		tmp = fscanf(f_kb4, "%ld", &validList_neg[i].r);
	}
	for (INT i = 0; i < testTotal_neg; i++) {
		tmp = fscanf(f_kb5, "%ld", &testList_neg[i].h);
		tmp = fscanf(f_kb5, "%ld", &testList_neg[i].t);
		tmp = fscanf(f_kb5, "%ld", &testList_neg[i].r);
	}

	fclose(f_kb1);
	fclose(f_kb2);
	fclose(f_kb3);
	fclose(f_kb4);
	fclose(f_kb5);

	std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
	std::sort(testList, testList + testTotal, Triple::cmp_rel2);
	std::sort(validList, validList + validTotal, Triple::cmp_rel2);
	std::sort(validList_neg, validList_neg + validTotal_neg, Triple::cmp_rel2);
	std::sort(testList_neg, testList_neg + testTotal_neg, Triple::cmp_rel2);

	testLef = (INT *)calloc(relationTotal, sizeof(INT));
	testRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(testLef, -1, sizeof(INT) * relationTotal);
	memset(testRig, -1, sizeof(INT) * relationTotal);
	for (INT i = 1; i < testTotal; i++) {
		if (testList[i].r != testList[i - 1].r) {
			testRig[testList[i - 1].r] = i - 1;
			testLef[testList[i].r] = i;
		}
	}
	testLef[testList[0].r] = 0;
	testRig[testList[testTotal - 1].r] = testTotal - 1;

	validLef = (INT *)calloc(relationTotal, sizeof(INT));
	validRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(validLef, -1, sizeof(INT) * relationTotal);
	memset(validRig, -1, sizeof(INT) * relationTotal);
	for (INT i = 1; i < validTotal; i++) {
		if (validList[i].r != validList[i - 1].r) {
			validRig[validList[i - 1].r] = i - 1;
			validLef[validList[i].r] = i;
		}
	}
	validLef[validList[0].r] = 0;
	validRig[validList[validTotal - 1].r] = validTotal - 1;

}

#endif
