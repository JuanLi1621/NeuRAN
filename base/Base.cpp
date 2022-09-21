#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include "Valid.h"
#include <cstdlib>
#include <pthread.h>
#include <time.h>
#include <string>

extern "C"
void setInPath(char *path);

extern "C"
void setOutPath(char *path);

// extern "C"
// void setTrainPath(char *path);

// extern "C"
// void setValidPath(char *path);

// extern "C"
// void setTestPath(char *path);

// extern "C"
// void setEntPath(char *path);

// extern "C"
// void setRelPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
INT getWorkThreads();

extern "C"
void setBern(INT con);

extern "C"
INT getHd_max();

extern "C"
INT getTl_max();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

// extern "C"
// INT getTrainPosTotal();

// extern "C"
// INT getTrainNoiTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;
	INT *batch_emb_h;
	INT *batch_emb_r;
    INT *batch_emb_t;
    REAL *batch_y;

    INT *batch_dm_nbrs;//neighbor relations of head (h,r/r1...) or tail (r/r1...,t)
    INT *batch_dm_nbrs_len;
    INT *batch_rg_nbrs;
    INT *batch_rg_nbrs_len;

    INT negRate;
	INT batchSize;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_emb_h = para -> batch_emb_h;
	INT *batch_emb_r = para -> batch_emb_r;
	INT *batch_emb_t = para -> batch_emb_t;
    REAL *batch_y = para -> batch_y;

	INT *batch_dm_nbrs = para -> batch_dm_nbrs;
	INT *batch_dm_nbrs_len = para -> batch_dm_nbrs_len;
	INT *batch_rg_nbrs = para -> batch_rg_nbrs;
	INT *batch_rg_nbrs_len = para -> batch_rg_nbrs_len;

    INT negRate = para -> negRate;
	INT batchSize = para -> batchSize;

	INT lef, rig;
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	for (INT batch = lef; batch < rig; batch++) {
		INT i = rand_max(id, trainTotal);
		batch_emb_h[batch] = trainList[i].h;
        batch_emb_t[batch] = trainList[i].t;
        batch_emb_r[batch] = trainList[i].r;
        batch_y[batch] = 1;
        
        batch_dm_nbrs_len[batch] = hd_nbrs_rig[trainList[i].h] - hd_nbrs_lef[trainList[i].h] + 1; //存储当前h连接的关系个数/关系集合
        INT n=0;
        //add (h,r) neighbors
        for (INT j = hd_nbrs_lef[trainList[i].h]; j <= hd_nbrs_rig[trainList[i].h]; j++){
            batch_dm_nbrs[batch * hd_max + n] = trainHdNbrs[j].r;
            n+=1;
        }
        for (INT c = n; c < hd_max; c++){
            batch_dm_nbrs[batch * hd_max + c] = relationTotal; //padding
        }
        batch_rg_nbrs_len[batch] = tl_nbrs_rig[trainList[i].t] - tl_nbrs_lef[trainList[i].t] + 1;
        INT m = 0;
        for (INT k = tl_nbrs_lef[trainList[i].t]; k <= tl_nbrs_rig[trainList[i].t]; k++){
            batch_rg_nbrs[batch * tl_max + m] = trainTlNbrs[k].r;
            m+=1;
        }
        for (INT c = m; c < tl_max; c++){
            batch_rg_nbrs[batch * tl_max + c] = relationTotal;
        }

		INT last = batchSize;
        for (INT times=0; times<negRate; times++){
            if (bernFlag)
                prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);

            //neg for kge module
            if (randd(id) % 1000 < prob) {
                //replace tail
                INT neg_emb_t = corrupt_head(id, trainList[i].h, trainList[i].r);           
                while (find_hrt(trainList[i].h, neg_emb_t, trainList[i].r) || (trainList[i].h==neg_emb_t)){
                    neg_emb_t = corrupt_head(id, trainList[i].h, trainList[i].r);
                }
                batch_emb_h[batch + last] = trainList[i].h;
                batch_emb_t[batch + last] = neg_emb_t;
                batch_emb_r[batch + last] = trainList[i].r;

                batch_dm_nbrs_len[batch + last] = hd_nbrs_rig[trainList[i].h] - hd_nbrs_lef[trainList[i].h] + 1; 
                INT n1=0;
                for (INT j = hd_nbrs_lef[trainList[i].h]; j <= hd_nbrs_rig[trainList[i].h]; j++){
                    batch_dm_nbrs[(batch+last)*hd_max + n1] =  trainHdNbrs[j].r;
                    n1+=1;
                }
                for (INT c = n1; c < hd_max; c++){
                    batch_dm_nbrs[(batch+last)*hd_max + c] = relationTotal;
                }
                if (tl_nbrs_lef[neg_emb_t]==0 && tl_nbrs_rig[neg_emb_t]==-1){//the replaced t does not has a neighbor
                    batch_rg_nbrs_len[batch + last] = 0;
                    batch_rg_nbrs[(batch+last)*tl_max] = trainList[i].r;
                    for (INT c = 1; c<tl_max; c++){
                        batch_rg_nbrs[(batch+last)*tl_max+c] = relationTotal;
                    }
                } else {
                    batch_rg_nbrs_len[batch + last] = tl_nbrs_rig[neg_emb_t] - tl_nbrs_lef[neg_emb_t] + 1;
                    INT m1 = 0;
                    for (INT k = tl_nbrs_lef[neg_emb_t]; k <= tl_nbrs_rig[neg_emb_t]; k++){
                        batch_rg_nbrs[(batch+last) * tl_max + m1] = trainTlNbrs[k].r;
                        m1+=1;
                    }
                    for (INT c = m1; c < tl_max; c++){
                        batch_rg_nbrs[(batch+last) * tl_max + c] = relationTotal;
                    }
                }
                batch_y[batch+last] = -1;
                last += batchSize;

            }else{
                 //replace head
                INT neg_emb_h = corrupt_tail(id, trainList[i].t, trainList[i].r);
                while (find_hrt(neg_emb_h, trainList[i].t, trainList[i].r)||trainList[i].t==neg_emb_h){
                    neg_emb_h = corrupt_tail(id, trainList[i].t, trainList[i].r);
                }
                batch_emb_h[batch + last] = neg_emb_h;
                batch_emb_t[batch + last] = trainList[i].t;
                batch_emb_r[batch + last] = trainList[i].r;

                if (hd_nbrs_lef[neg_emb_h]==0 && hd_nbrs_rig[neg_emb_h]==-1){
                    batch_dm_nbrs_len[batch + last] = 0;
                    batch_dm_nbrs[(batch + last) * hd_max] = trainList[i].r;
                    for (INT c=1; c<hd_max; c++){
                        batch_dm_nbrs[(batch + last) * hd_max + c] = relationTotal;
                    }
                } else {
                    batch_dm_nbrs_len[batch + last] = hd_nbrs_rig[neg_emb_h] - hd_nbrs_lef[neg_emb_h] + 1;
                    INT n1=0;
                    for (INT j = hd_nbrs_lef[neg_emb_h]; j <= hd_nbrs_rig[neg_emb_h]; j++){
                        batch_dm_nbrs[(batch+last) * hd_max + n1] = trainHdNbrs[j].r;
                        n1+=1;
                    }
                    for (INT c = n1; c < hd_max; c++){
                    batch_dm_nbrs[(batch + last) * hd_max + c] = relationTotal;
                    }
                }                    

                batch_rg_nbrs_len[batch + last] = tl_nbrs_rig[trainList[i].t] - tl_nbrs_lef[trainList[i].t] + 1;
                INT m1 = 0;
                for (INT k = tl_nbrs_lef[trainList[i].t]; k <= tl_nbrs_rig[trainList[i].t]; k++){
                    batch_rg_nbrs[(batch+last) * tl_max + m1] = trainTlNbrs[k].r;
                    m1+=1;
                }
                for (INT c = m1; c < tl_max; c++){
                    batch_rg_nbrs[(batch+last) * tl_max + c] = relationTotal;
                } 
                batch_y[batch+last] = -1; 
                last += batchSize;
            }
        }
    }
	pthread_exit(NULL);
}

extern "C"
void sampling(INT *batch_emb_h, INT *batch_emb_t, INT *batch_emb_r, REAL *batch_y, INT batchSize, INT negRate, 
    INT *batch_dm_nbrs, INT *batch_dm_nbrs_len, INT *batch_rg_nbrs, INT *batch_rg_nbrs_len) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_emb_h = batch_emb_h;
		para[threads].batch_emb_t = batch_emb_t;
		para[threads].batch_emb_r = batch_emb_r;
        para[threads].batch_y = batch_y;

        para[threads].batch_dm_nbrs = batch_dm_nbrs;
		para[threads].batch_dm_nbrs_len = batch_dm_nbrs_len;
        para[threads].batch_rg_nbrs = batch_rg_nbrs;
		para[threads].batch_rg_nbrs_len = batch_rg_nbrs_len;

		para[threads].batchSize = batchSize;
        para[threads].negRate = negRate;

		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);
	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}
