#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot=0, l1_tot=0, r1_filter_tot=0, r1_tot=0, l_tot=0, r_tot=0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;

extern "C"
void initTest(){
    lastHead = 0;
    lastTail = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
}

extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr, INT *p_dm_nbr, INT *p_dm_nbr_len, INT *p_rg_nbr, INT *p_rg_nbr_len){ //INT *p_dm_nbr_prob, INT *p_rg_nbr_prob
    for (INT i=0; i<entityTotal; i++){
        ph[i] = i;
        pt[i] = testList[lastHead].t;
        pr[i] = testList[lastHead].r;
        if (hd_nbrs_lef[i]==0 && hd_nbrs_rig[i]==-1){
            p_dm_nbr_len[i] = 0;
            p_dm_nbr[i*hd_max] = testList[lastHead].r;
            for (INT c = 1; c < hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            p_dm_nbr_len[i] = hd_nbrs_rig[i] - hd_nbrs_lef[i] + 1;
            INT m = 0;
            for (INT j=hd_nbrs_lef[i]; j<=hd_nbrs_rig[i]; j++){
                p_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r;
                m+=1;
            }
            for (INT c = m; c < hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }
        
        if (tl_nbrs_lef[testList[lastHead].t]==0 && tl_nbrs_rig[testList[lastHead].t]==-1){
            p_rg_nbr_len[i] = 0;
            p_rg_nbr[i*tl_max] = testList[lastHead].r;
            for (INT c=1; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            p_rg_nbr_len[i] = tl_nbrs_rig[testList[lastHead].t] - tl_nbrs_lef[testList[lastHead].t]+1;
            INT n=0;
            for (INT j=tl_nbrs_lef[testList[lastHead].t]; j<=tl_nbrs_rig[testList[lastHead].t]; j++){
                p_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r;
                n+=1;
            }
            for (INT c=n; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }
    }
}

extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr, INT *p_dm_nbr, INT *p_dm_nbr_len, INT *p_rg_nbr, INT *p_rg_nbr_len){
    for (INT i=0; i<entityTotal; i++){
        ph[i] = testList[lastTail].h;
        pt[i] = i;
        pr[i] = testList[lastTail].r;
        if (hd_nbrs_lef[testList[lastTail].h]==0 && hd_nbrs_rig[testList[lastTail].h]==-1){
            p_dm_nbr_len[i] = 0;
            p_dm_nbr[i*hd_max] = testList[lastTail].r;
            for (INT c = 1; c < hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            p_dm_nbr_len[i] = hd_nbrs_rig[testList[lastTail].h] - hd_nbrs_lef[testList[lastTail].h] + 1;
            INT m = 0;
            for (INT j=hd_nbrs_lef[testList[lastTail].h]; j<=hd_nbrs_rig[testList[lastTail].h]; j++){
                p_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r;
                m+=1;
            }
            for (INT c = m; c < hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }

        if (tl_nbrs_lef[i]==0 && tl_nbrs_rig[i]==-1){
            p_rg_nbr_len[i] = 0;
            p_rg_nbr[i*tl_max] = testList[lastTail].r;
            for (INT c=1; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            p_rg_nbr_len[i] = tl_nbrs_rig[i] - tl_nbrs_lef[i]+1;
            INT n=0;
            for (INT j=tl_nbrs_lef[i]; j<=tl_nbrs_rig[i]; j++){
                p_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r;
                n+=1;
            }
            for (INT c=n; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }
    }
}

extern "C"
void testHead(REAL *con){
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;
    REAL minimal = con[h];
    INT l_s = 0;
    INT l_filter_s = 0;
    for (INT j=0; j<entityTotal; j++){
        if (j!=h){
            REAL value = con[j];
            if (value < minimal){
                l_s+=1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
        }
    }
    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1+l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);

    lastHead++;
}

extern "C"
void testTail(REAL *con){
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    REAL minimal = con[t];
    INT r_s = 0;
    INT r_filter_s = 0;
    for (INT j = 0; j < entityTotal; j++){
        if (j!=t){
            REAL value = con[j];
            if (value < minimal){
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
        }
    }
    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);

    lastTail++;
}

extern "C"
void test_link_prediction(){
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;

    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;

    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;

    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;

    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    printf("link prediction results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);
}


//noise detection
extern "C"
void getNoiPosBatch(INT *noi_emb_h, INT *noi_emb_t, INT *noi_emb_r, INT *noi_dm_nbr, INT *noi_dm_nbr_len, INT *noi_rg_nbr, INT *noi_rg_nbr_len){
   printf("start getNoiPosBatch\n");
   printf("trainPosTotal=%ld\n", trainPosTotal);
   for (INT i = 0; i < trainPosTotal; i++){
        //emb
       noi_emb_h[i] = trainPosList[i].h;
       noi_emb_t[i] = trainPosList[i].t;
       noi_emb_r[i] = trainPosList[i].r;
       //dm
       if (hd_nbrs_lef[trainPosList[i].h]==0 && hd_nbrs_rig[trainPosList[i].h]==-1){
           printf("error occurs!");
       }else{
           noi_dm_nbr_len[i] = hd_nbrs_rig[trainPosList[i].h] - hd_nbrs_lef[trainPosList[i].h] + 1;
           INT m = 0;
           for (INT j = hd_nbrs_lef[trainPosList[i].h]; j <= hd_nbrs_rig[trainPosList[i].h]; j++){
               noi_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r;
               m+=1;
           }
           for (INT c = m; c < hd_max; c++){
               noi_dm_nbr[i*hd_max+c] = relationTotal;
           }
       }
       //rg
       if (tl_nbrs_lef[trainPosList[i].t]==0 && tl_nbrs_rig[trainPosList[i].t]==-1){
           printf("error occurs!");
       }else{
           noi_rg_nbr_len[i] = tl_nbrs_rig[trainPosList[i].t] - tl_nbrs_lef[trainPosList[i].t] + 1;
           INT n=0;
           for (INT j = tl_nbrs_lef[trainPosList[i].t]; j <= tl_nbrs_rig[trainPosList[i].t]; j++){
               noi_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r; //
               n+=1;
           }
           for (INT c = n; c<tl_max; c++){
               noi_rg_nbr[i*tl_max+c] = relationTotal;
           }
       }
   }
}

extern "C"
void getNoiNegBatch(INT *noi_neg_emb_h, INT *noi_neg_emb_t, INT *noi_neg_emb_r, INT *noi_neg_dm_nbr, INT *noi_neg_dm_nbr_len, INT *noi_neg_rg_nbr, INT *noi_neg_rg_nbr_len){
   printf("start getNoiNegBatch\n");
   printf("trainNoiTotal=%ld\n", trainNoiTotal);
   for (INT i = 0; i < trainNoiTotal; i++){
        //emb
       noi_neg_emb_h[i] = trainNoiList[i].h;
       noi_neg_emb_t[i] = trainNoiList[i].t;
       noi_neg_emb_r[i] = trainNoiList[i].r;
       //dm
       if (hd_nbrs_lef[trainNoiList[i].h]==0 && hd_nbrs_rig[trainNoiList[i].h]==-1){
           printf("error occurs!");
       }else{
           noi_neg_dm_nbr_len[i] = hd_nbrs_rig[trainNoiList[i].h] - hd_nbrs_lef[trainNoiList[i].h] + 1;
           INT m = 0;
           for (INT j = hd_nbrs_lef[trainNoiList[i].h]; j <= hd_nbrs_rig[trainNoiList[i].h]; j++){
               noi_neg_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r;
               m+=1;
           }
           for (INT c = m; c < hd_max; c++){
               noi_neg_dm_nbr[i*hd_max+c] = relationTotal;
           }
       }
       //rg
       if (tl_nbrs_lef[trainNoiList[i].t]==0 && tl_nbrs_rig[trainNoiList[i].t]==-1){
           printf("error occurs!");
       }else{
           noi_neg_rg_nbr_len[i] = tl_nbrs_rig[trainNoiList[i].t] - tl_nbrs_lef[trainNoiList[i].t] + 1;
           INT n=0;
           for (INT j = tl_nbrs_lef[trainNoiList[i].t]; j <= tl_nbrs_rig[trainNoiList[i].t]; j++){
               noi_neg_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r; //
               n+=1;
           }
           for (INT c = n; c<tl_max; c++){
               noi_neg_rg_nbr[i*tl_max+c] = relationTotal;
           }
       }
   }
}


/*=====================================================================================
triple classification
======================================================================================*/

Triple *negValidList;
extern "C"
void getNegValid() {
    negValidList = (Triple *)calloc(validTotal_neg, sizeof(Triple));
    for (INT i = 0; i < validTotal_neg; i++) {
        negValidList[i] = validList_neg[i];
    }   
}

Triple *negTestList;
extern "C"
void getNegTest() {
    negTestList = (Triple *)calloc(testTotal_neg, sizeof(Triple));
    for (INT i = 0; i < testTotal_neg; i++) {
        negTestList[i] = testList_neg[i];
    }
}

extern "C"
void getTestBatch(INT *pos_emb_h, INT *pos_emb_t, INT *pos_emb_r, INT *pos_dm_nbr, INT *pos_dm_nbr_len, INT *pos_rg_nbr, INT *pos_rg_nbr_len,  //pos
                INT *neg_emb_h, INT *neg_emb_t, INT *neg_emb_r, INT *neg_dm_nbr, INT *neg_dm_nbr_len, INT *neg_rg_nbr, INT *neg_rg_nbr_len){
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        pos_emb_h[i] = testList[i].h;
        pos_emb_t[i] = testList[i].t;
        pos_emb_r[i] = testList[i].r;
        //for dm_nbr
        if (hd_nbrs_lef[testList[i].h] == 0 && hd_nbrs_rig[testList[i].h] == -1){
            pos_dm_nbr_len[i] = 0;
            pos_dm_nbr[i*hd_max] = testList[i].r;
            for (INT c = 1; c<hd_max; c++){
                pos_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            pos_dm_nbr_len[i] = hd_nbrs_rig[testList[i].h] - hd_nbrs_lef[testList[i].h] + 1;
            INT m=0;
            for (INT j = hd_nbrs_lef[testList[i].h]; j <= hd_nbrs_rig[testList[i].h]; j++){
                pos_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r; //
                m+=1;
            }
            for (INT c = m; c<hd_max; c++){
                pos_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }
        //for rg_nbr
        if (tl_nbrs_lef[testList[i].t]==0 && tl_nbrs_rig[testList[i].t]==-1){
            pos_rg_nbr_len[i] = 0;
            pos_rg_nbr[i*tl_max] = testList[i].r;
            for (INT c = 1; c<tl_max; c++){
                pos_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            pos_rg_nbr_len[i] = tl_nbrs_rig[testList[i].t] - tl_nbrs_lef[testList[i].t] + 1;
            INT n=0;
            for (INT j = tl_nbrs_lef[testList[i].t]; j <= tl_nbrs_rig[testList[i].t]; j++){
                pos_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r; //
                n+=1;
            }
            for (INT c = n; c<tl_max; c++){
                pos_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }
    }

    for (INT i = 0; i < testTotal_neg; i++){//negative triple for random replace h or t
        neg_emb_h[i] = negTestList[i].h;
        neg_emb_t[i] = negTestList[i].t;
        neg_emb_r[i] = negTestList[i].r;

        if (hd_nbrs_lef[negTestList[i].h]==0 && hd_nbrs_rig[negTestList[i].h]==-1){
            neg_dm_nbr_len[i] = 0;
            neg_dm_nbr[i*hd_max] = negTestList[i].r;
            for (INT c = 1; c<hd_max; c++){
                neg_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            neg_dm_nbr_len[i] = hd_nbrs_rig[negTestList[i].h] - hd_nbrs_lef[negTestList[i].h] + 1;
            INT x=0;
            for (INT j = hd_nbrs_lef[negTestList[i].h]; j <= hd_nbrs_rig[negTestList[i].h]; j++){
                neg_dm_nbr[i*hd_max+x] = trainHdNbrs[j].r; //
                x+=1;
            }
            for (INT c = x; c<hd_max; c++){
                neg_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }
        if (tl_nbrs_lef[negTestList[i].t]==0 && tl_nbrs_rig[negTestList[i].t]==-1){
            neg_rg_nbr_len[i] = 0;
            neg_rg_nbr[i*tl_max] = negTestList[i].r;
            for (INT c = 1; c<tl_max; c++){
                neg_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            neg_rg_nbr_len[i] = tl_nbrs_rig[negTestList[i].t] - tl_nbrs_lef[negTestList[i].t] + 1;
            INT y=0;
            for (INT j = tl_nbrs_lef[negTestList[i].t]; j <= tl_nbrs_rig[negTestList[i].t]; j++){
                neg_rg_nbr[i*tl_max+y] = trainTlNbrs[j].r; //
                y+=1;
            }
            for (INT c = y; c<tl_max; c++){
                neg_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }
    }
}

extern "C"
void getValidBatch(INT *p_emb_h, INT *p_emb_t, INT *p_emb_r, INT *p_dm_nbr, INT *p_dm_nbr_len, INT *p_rg_nbr, INT *p_rg_nbr_len, //pos
                INT *n_emb_h, INT *n_emb_t, INT *n_emb_r, INT *n_dm_nbr, INT *n_dm_nbr_len, INT *n_rg_nbr, INT *n_rg_nbr_len){
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        p_emb_h[i] = validList[i].h;
        p_emb_t[i] = validList[i].t;
        p_emb_r[i] = validList[i].r;
        //for dm_nbr
        if (hd_nbrs_lef[validList[i].h]==0 && hd_nbrs_rig[validList[i].h]==-1){
            p_dm_nbr_len[i] = 0;
            p_dm_nbr[i*hd_max] = validList[i].r;
            for (INT c = 1; c<hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            p_dm_nbr_len[i] = hd_nbrs_rig[validList[i].h] - hd_nbrs_lef[validList[i].h] + 1;
            INT m=0;
            for (INT j = hd_nbrs_lef[validList[i].h]; j <= hd_nbrs_rig[validList[i].h]; j++){
                p_dm_nbr[i*hd_max+m] = trainHdNbrs[j].r; //
                m+=1;
            }
            for (INT c = m; c<hd_max; c++){
                p_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }
        //for rg_nbr
        if (tl_nbrs_lef[validList[i].t]==0 && tl_nbrs_rig[validList[i].t]==-1){
            p_rg_nbr_len[i] = 0;
            p_rg_nbr[i*tl_max] = validList[i].r;
            for (INT c = 1; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            p_rg_nbr_len[i] = tl_nbrs_rig[validList[i].t] - tl_nbrs_lef[validList[i].t] + 1;
            INT n=0;
            for (INT j = tl_nbrs_lef[validList[i].t]; j <= tl_nbrs_rig[validList[i].t]; j++){
                p_rg_nbr[i*tl_max+n] = trainTlNbrs[j].r; //
                n+=1;
            }
            for (INT c = n; c<tl_max; c++){
                p_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }
    }

    for (INT i = 0; i < validTotal_neg; i++) {
        n_emb_h[i] = negValidList[i].h;
        n_emb_t[i] = negValidList[i].t;
        n_emb_r[i] = negValidList[i].r;
        //for neg_dm_nbr
        if (hd_nbrs_lef[negValidList[i].h]==0 && hd_nbrs_rig[negValidList[i].h]==-1){
            n_dm_nbr_len[i] = 0;
            n_dm_nbr[i*hd_max] = negValidList[i].r;
            for (INT c = 1; c<hd_max; c++){
                n_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }else{
            n_dm_nbr_len[i] = hd_nbrs_rig[negValidList[i].h] - hd_nbrs_lef[negValidList[i].h] + 1;
            INT x=0;
            for (INT j = hd_nbrs_lef[negValidList[i].h]; j <= hd_nbrs_rig[negValidList[i].h]; j++){
                n_dm_nbr[i*hd_max+x] = trainHdNbrs[j].r; //
                x+=1;
            }
            for (INT c = x; c<hd_max; c++){
                n_dm_nbr[i*hd_max+c] = relationTotal;
            }
        }
        //for neg_rg_nbr 
        if (tl_nbrs_lef[negValidList[i].t]==0 && tl_nbrs_rig[negValidList[i].t]==-1){
            n_rg_nbr_len[i] = 0;
            n_rg_nbr[i*tl_max] = negValidList[i].r;
            for (INT c = 1; c<tl_max; c++){
                n_rg_nbr[i*tl_max+c] = relationTotal;
            }
        }else{
            n_rg_nbr_len[i] = tl_nbrs_rig[negValidList[i].t] - tl_nbrs_lef[negValidList[i].t] + 1;
            INT y=0;
            for (INT j = tl_nbrs_lef[negValidList[i].t]; j <= tl_nbrs_rig[negValidList[i].t]; j++){
                n_rg_nbr[i*tl_max+y] = trainTlNbrs[j].r; //
                y+=1;
           }
           for (INT c = y; c<tl_max; c++){
               n_rg_nbr[i*tl_max+c] = relationTotal;
           }
        }
    }
}

REAL threshEntire;
extern "C"
void getBestThreshold(REAL *relThresh, REAL *score_pos, REAL *score_neg) {
    REAL interval = 0.0005;
    REAL min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total, tmp;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        total = (validRig[r] - validLef[r] + 1) * 2;
        min_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] < min_score) min_score = score_neg[validLef[r]];
        max_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] > max_score) max_score = score_neg[validLef[r]];
        for (INT i = validLef[r]+1; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        n_interval = INT((max_score - min_score)/interval);
        for (INT i = 0; i <= n_interval; i++) {
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] <= tmpThresh) correct ++;
                if (score_neg[j] > tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
            }
        }
        relThresh[r] = bestThresh;
    }
}
REAL *testAcc;
REAL aveAcc;
extern "C"
void test_triple_classification(REAL *relThresh, REAL *score_pos, REAL *score_neg) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    REAL aveAcc;
    INT tp_score=0, fn_score=0, tn_score=0, fp_score=0, tmp;
    REAL precision_tc, recall_tc;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]){
                correct++;
                tp_score++;
            }else{
                fn_score++;
            }
            if (score_neg[i] >relThresh[r]){
                correct++;
                tn_score++;
            }else{
                fp_score++;
            }
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct; 
        aveTotal += total;
    }
    printf("total test triples = %ld\n", aveTotal/2);
    aveAcc = 1.0 * aveCorrect / aveTotal;
    precision_tc = tp_score*1.0/(tp_score+fp_score);
    recall_tc = tp_score*1.0/(tp_score+fn_score);
    printf("tp=%ld, fn=%ld, fp=%ld, tn=%ld\n", tp_score, fn_score, fp_score, tn_score);
    printf("triple classification accuracy is %lf, precision is %lf, recall is %lf\n", aveAcc, precision_tc, recall_tc);
}


#endif
