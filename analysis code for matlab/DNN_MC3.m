%% DNN_MC3.m 
%% Update; 240308
%% usage; peak50ms: N x 2001 matrix(for vitro),N x 4001 matrix(for vivo)  
%% modulate LFP trace → as x_test(after processing python)
% peak50ms: extracted 50 ms before and after SW peak from atf file
% scaling in the range of [0-1]（210311）

peak50ms_n=normalize(peak50ms,2,'range');

csvwrite('slice2_5cells_102.txt',peak50ms_n);% <---- modify


%% modulate vm trace

peak50ms_vm1=peak50ms_vm(:,:,1);
peak50ms_vm2=peak50ms_vm(:,:,2);
peak50ms_vm3=peak50ms_vm(:,:,3);
peak50ms_vm4=peak50ms_vm(:,:,4);
peak50ms_vm5=peak50ms_vm(:,:,5);

peak50ms_vm1_n=normalize(peak50ms_vm1,2,'range');
peak50ms_vm2_n=normalize(peak50ms_vm2,2,'range');
peak50ms_vm3_n=normalize(peak50ms_vm3,2,'range');
peak50ms_vm4_n=normalize(peak50ms_vm4,2,'range');
peak50ms_vm5_n=normalize(peak50ms_vm5,2,'range');

peak50ms_vm_n=cat(3,peak50ms_vm1_n,peak50ms_vm2_n,peak50ms_vm3_n,peak50ms_vm4_n,peak50ms_vm5_n);
csvwrite('slice2_vmtrace_normalized_102.txt',peak50ms_vm_n); % <---- modify


%% create random numbers for shuffle
for j=1:size(peak50ms_vm,3)
    for i=1:100
        R(:,i,j)=randperm(size(peak50ms_vm,1));
    end
end
csvwrite('slice2_Vmshuffleno_102.txt',R); % <---- modify

%% load output waveform learning with python→x_test

csvfiles=dir('*.csv');% be careful for the PATH!!
numfiles=length(csvfiles);
X_test=cell(1,numfiles);

for k=1:numfiles
    X_test{k}=importdata(csvfiles(k).name);
end
clear k numfiles

%% processing after training with python ## k-fold cross validation ##
% 210630
% Decoded_vm; output SW waveform
% X_test; original SW waveform (scaled 0-1)
% above two variables stored as 1×k cell（k = 5 in this cae）

% Average of the bottom 10-20% of LFP source data（①）
for j = 1 : size(X_test,2)
    for i = 1 : size(X_test{1,1},1)
        LFP_sort{:,j}(i,:) = sort(X_test{:,j}(i,:));
        LFP_sort_10_20{:,j}(i,:) = LFP_sort{:,j}(i,200:400);
        LFP_sort_mean{:,j}(i,:) = mean(LFP_sort_10_20{:,j}(i,:),2);
    end
end
clear i j

% Average of the bottom 10-20% of decoded SW waveform（②）
for j = 1 : size(Decoded_vm,2)
    for i = 1 : size(Decoded_vm{1,1},1)
        vmsort{:,j}(i,:) = sort(Decoded_vm{:,j}(i,:));
        vmsort_10_20{:,j}(i,:) = vmsort{:,j}(i,200:400);
        vmsort_mean{:,j}(i,:)= mean(vmsort_10_20{:,j}(i,:),2);
    end
end
clear i j

% correct baseline（② - ①）
vmsort_double = cell2mat(vmsort_mean);
LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_double - LFPsort_double;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(Decoded_vm,2)
    m_LFP(:,i) = {Decoded_vm{1,i} - base_corr_comb(:,i)};
end
clear i

% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for j=1:size(Decoded_vm,2)
    for i=1:size(X_test{1,1},1)
        RMSE_corr_cv{:,j}(i,:) = sqrt(immse(X_test{:,j}(i,:),m_LFP{:,j}(i,:)));
    end
end
clear i j
RMSE_corr_cv = cell2mat(RMSE_corr_cv);

RMSE_corr_cv_m = mean(RMSE_corr_cv,2);

%% For each combination、after shuffling ## import data ##
% calculate RMSE in each combination with correcting baseling
% usage; decoded SW waveform: size(x_test,1)×2001×N（or 1×N cell）
% 上述のLFPsort_mean_2は固定される
clear csvfiles numfiles

% import data
csvfiles=dir('*.csv');%% be careful for the PATH!!
numfiles=length(csvfiles);
decoded_vm=cell(1,numfiles);

for k=1:numfiles
    decoded_vm{k}=importdata(csvfiles(k).name);
end
clear k numfiles
%% For each combination、after shuffling　## import data ##
% calculate RMSE in each combination with correcting baseling
% usage; decoded SW waveform: size(x_test,1)×2001×N（or 1×N cell）
% 上述のLFPsort_mean_2は固定される
clear csvfiles numfiles

% import data
csvfiles=dir('*.csv');% be careful for the PATH!!
numfiles=length(csvfiles);
x_test=cell(1,numfiles);

for k=1:numfiles
    x_test{k}=importdata(csvfiles(k).name);
end
clear k numfiles
%% For each combination、after shuffling　## correct baseline & calculate RMSE ##
% ## k-fold cross-validation ##
% 5-fold cv、10-times shuffle each in python（total 50 of x_test and decoded_vm_s）
clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
clear LFP_sort LFP_sort_10_20 LFP_sort_mean combset shufflenum

combset = 3; % number of combinations <= modify
shufflenum = 10; % 1 cell prediction→100、other→10

% Average of the bottom 10-20% of LFP source data（①）
for j = 1 : size(x_test,2)
    for i = 1 : size(x_test{1,1},1)
        LFP_sort{:,j}(i,:) = sort(x_test{:,j}(i,:));
        LFP_sort_10_20{:,j}(i,:) = LFP_sort{:,j}(i,200:400);
        LFP_sort_mean{:,j}(i,:) = mean(LFP_sort_10_20{:,j}(i,:),2);
    end
end
clear i j

% Average of the bottom 10-20% of decoded SW waveform（②）
for j = 1 : size(decoded_vm,2)
    for i = 1 : size(decoded_vm{1,1},1)
        vmsort{:,j}(i,:) = sort(decoded_vm{:,j}(i,:));
        vmsort_10_20{:,j}(i,:) = vmsort{:,j}(i,200:400);
        vmsort_mean{:,j}(i,:)= mean(vmsort_10_20{:,j}(i,:),2);
    end
end
clear i j

% correct baseline（② - ①）
vmsort_double = cell2mat(vmsort_mean);
LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_double - LFPsort_double;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(decoded_vm,2)
    m_LFP_s(:,i) = {decoded_vm{1,i} - base_corr_comb(:,i)};
end
clear i

% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for j=1:size(decoded_vm,2)
    for i=1:size(x_test{1,1},1)
        RMSE_corr_cv_s{:,j}(i,:) = sqrt(immse(x_test{:,j}(i,:),m_LFP_s{:,j}(i,:)));
    end
end
clear i j

RMSE_corr_cv_s = cell2mat(RMSE_corr_cv_s);
RMSE_corr_cv_s = reshape(RMSE_corr_cv_s,[size(base_corr_comb,1)*5,combset*shufflenum]);

% RMSE_corr_cv_s_m = mean(RMSE_corr_cv_s,2);

%% For each combination、after shuffling　## correct baseline & calculate RMSE ##
% #### shuffle-cvは取り込み方に注意 ####
% ## k-fold cross-validation ##
%  5-fold cv、10-times shuffle each in python（total 50 of x_test and decoded_vm_s）
clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
clear LFP_sort LFP_sort_10_20 LFP_sort_mean combset shufflenum
% 
% combset = 1; % number of combinations
% shufflenum = 10; % 1 cell predictionのときは100、それ以外は10！

% Average of the bottom 10-20% of LFP source data（①）
for j = 1 : size(x_test,2)
    for i = 1 : size(x_test{1,1},1)
        LFP_sort{:,j}(i,:) = sort(x_test{:,j}(i,:));
        LFP_sort_10_20{:,j}(i,:) = LFP_sort{:,j}(i,200:400);
        LFP_sort_mean{:,j}(i,:) = mean(LFP_sort_10_20{:,j}(i,:),2);
    end
end
clear i j

% Average of the bottom 10-20% of decoded SW waveform（②）
for j = 1 : size(decoded_vm,2)
    for i = 1 : size(decoded_vm{1,1},1)
        vmsort{:,j}(i,:) = sort(decoded_vm{:,j}(i,:));
        vmsort_10_20{:,j}(i,:) = vmsort{:,j}(i,200:400);
        vmsort_mean{:,j}(i,:)= mean(vmsort_10_20{:,j}(i,:),2);
    end
end
clear i j

% correct baseline（② - ①）
vmsort_double = cell2mat(vmsort_mean);
LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_double - LFPsort_double;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(decoded_vm,2)
    m_LFP_s(:,i) = {decoded_vm{1,i} - base_corr_comb(:,i)};
end
clear i

% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for j=1:size(decoded_vm,2)
    for i=1:size(x_test{1,1},1)
        RMSE_corr_cv_s{:,j}(i,:) = sqrt(immse(x_test{:,j}(i,:),m_LFP_s{:,j}(i,:)));
    end
end
clear i j

RMSE_corr_cv_s = cell2mat(RMSE_corr_cv_s);

for i = 1:10
    % RMSE_corr_cv_Re(:,1)が各cvにおける（つまり全SW）シャッフル1回目
    RMSE_corr_cv_Re(:,i) = [RMSE_corr_cv_s(:,i);RMSE_corr_cv_s(:,i+10);RMSE_corr_cv_s(:,i+20);RMSE_corr_cv_s(:,i+30);RMSE_corr_cv_s(:,i+40)];
end
clear i
clear RMSE_corr_cv_s
RMSE_corr_cv_s = RMSE_corr_cv_Re;
clear RMSE_corr_cv_Re

RMSE_corr_cv_s_m = mean(RMSE_corr_cv_s,2);

%% For each combination、after shuffling　## correct baseline & calculate RMSE ##
% ## k-fold cross-validation ##
% 5-fold cv、10-times shuffle each in python（total 50 of x_test and decoded_vm_s）
clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
clear LFP_sort LFP_sort_10_20 LFP_sort_mean

combset = 5;

% Average of the bottom 10-20% of LFP source data（①）
for j = 1 : size(x_test,2)
    for i = 1 : size(x_test{1,1},1)
        LFP_sort{:,j}(i,:) = sort(x_test{:,j}(i,:));
        LFP_sort_10_20{:,j}(i,:) = LFP_sort{:,j}(i,200:400);
        LFP_sort_mean{:,j}(i,:) = mean(LFP_sort_10_20{:,j}(i,:),2);
    end
end
clear i j

% Average of the bottom 10-20% of decoded SW waveform（②）
for j = 1 : size(decoded_vm,2)
    for i = 1 : size(decoded_vm{1,1},1)
        vmsort{:,j}(i,:) = sort(decoded_vm{:,j}(i,:));
        vmsort_10_20{:,j}(i,:) = vmsort{:,j}(i,200:400);
        vmsort_mean{:,j}(i,:)= mean(vmsort_10_20{:,j}(i,:),2);
    end
end
clear i j

% correct baseline（② - ①）
vmsort_double = cell2mat(vmsort_mean);
LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_double - LFPsort_double;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(decoded_vm,2)
    m_LFP_s(:,i) = {decoded_vm{1,i} - base_corr_comb(:,i)};
end
clear i


% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for j=1:size(decoded_vm,2)
    for i=1:size(x_test{1,1},1)
        RMSE_corr_cv{:,j}(i,:) = sqrt(immse(x_test{:,j}(i,:),m_LFP_s{:,j}(i,:)));
    end
end
clear i j


RMSE_corr_cv = cell2mat(RMSE_corr_cv);
RMSE_corr_cv = reshape(RMSE_corr_cv,[size(base_corr_comb,1)*5,combset]);

% RMSE_corr_cv_m = mean(RMSE_corr_cv,2);

%% hilus map

% update;240118
sizex=[1:1:87]';
h=imagesc('CData',sizex);
CData=h.CData;
cmap = jet(256);
cmin=min(CData(:));
cmax=max(CData(:));
m=length(cmap);
index = fix((CData-cmin)/(cmax-cmin)*m)+1;
RGB=ind2rgb(index,cmap);
RGB_stack = [RGB(:,:,1) RGB(:,:,2) RGB(:,:,3)];
% RGB_stack_c =RGB_stack .* 255;% for PP

% for check
% x=1:1:87;
% y=1:1:87;
% figure;gscatter(x,y,x,RGB_stack);

% import D value
[~,ind] = sort(prediction_rate_1cell);
R = deg2rad(norm_angle(ind));
rho = norm_dist(ind);


% fig4B
figure;
for i=1:size(R,1)
pl = polarplot(R(i,:),rho(i,:),'.');thetalim([0 180]);rlim([0 1]);hold on;
pl.Color = RGB_stack(i,:);
pl.MarkerSize = 12;
end

% figure;
% for i=1:10
% pl = polarplot(R(i,:),rho(i,:),'.');thetalim([0 180]);rlim([0 1]);hold on;
% pl.Color = color2(i,:);
% pl.MarkerSize = 12;
% end

% correlation coefficient of the RMSE scores between MC pairs 
clear v C
v=1:3; C=nchoosek(v,2); % v=1:4; v=1:5;

for i=1:size(C,1)
    rr{i}=corrcoef(RMSE_judge_3c1(:,C(i,1)),RMSE_judge_3c1(:,C(i,2)));% RMSE_judge_4c1,RMSE_judge_5c1
    r(i,:)=rr{1,i}(1,2);
end
clear i
r_all = [r_all;r]; %r_all_fix...from 3celldata ⇒ 5celldata
clearvars -except r_all

% figure out relationship between corrcoef and cell-cell distance
v=1:3; C=nchoosek(v,2);
CC=[C;C+3;C+6;C+9;C+12;C+15;C+18;C+21;C+24;C+27;C+30;C+33;C+36;C+39;C+42];
clear v C
v=46:49; C=nchoosek(v,2);
CC=[CC;C;C+4;C+8;C+12;C+16;C+20;C+24;C+28];
clear v C
v=78:82; C=nchoosek(v,2);
CC=[CC;C;C+5];% cell ID of total corrcoef

[xx,yy]=pol2cart(R,rho); %極座標から直交座標xyに変換
RRR=[xx yy];

cellpdist = pdist(RRR);
cellpdist_Z = squareform(cellpdist);

for i=1:size(CC,1)
    corr(i)=cellpdist_Z(CC(i,:),CC(1,2));
end
corr=corr';

[r,p] = corrcoef(r_all_fix,corr)

% RR=[deg2rad(norm_angle) norm_dist];　% this variable doesn't work when use
% for knnsearch

% fig4C --spatial entropy
[Idx,~] = knnsearch(RRR,RRR,'K',5);% get 4 points around, 5 points including own point.default=euclidean distance
D_val = prediction_rate_1cell(Idx);
ch_rate1_sum = sum(D_val,2);
D_val_sumsum = sum(ch_rate1_sum);
P = ch_rate1_sum ./ D_val_sumsum;
H = -(P.*log2(P));
H_sum = sum(H); % real entropy



% shuffle Idx（N×5）（N is number of x_test）the most left side is fixed. Strictly shuffle of N x 4
I = [1:size(D,1)]';
for j=1:1e4
    for i=1:size(D,1)
        Idx_s(i,:) =randperm(size(D,1),4);
    end
    clear i
    Idx_s =[I Idx_s];

    for i=1:size(D,1)
    
        if Idx_s(i,1) == Idx_s(i,2)
            Idx_s(i,2) = randperm(size(D,1),1);
        elseif Idx_s(i,1) == Idx_s(i,3)
            Idx_s(i,3) = randperm(size(D,1),1);
        elseif Idx_s(i,1) == Idx_s(i,4)
            Idx_s(i,4) = randperm(size(D,1),1);
        elseif Idx_s(i,1) == Idx_s(i,5)
            Idx_s(i,5) = randperm(size(D,1),1);
        end
        
    end

D_val_s = prediction_rate_1cell(Idx_s);
co_sum_s = sum(D_val_s,2);
co_sumsum_s = sum(co_sum_s);
P_s = co_sum_s ./ co_sumsum_s;
H_s = -(P_s.*log2(P_s)); 
H_sum_s(j,:)=sum(H_s); % shuffle entropy
clear D_val_s co_sum_s co_sumsum_s P_s H_s Idx_s
end
entropy_s_sort=sort(H_sum_s);
entropy_s_inf95=entropy_s_sort(1e4*0.05+1);


% fig.4C
figure;histogram(H_sum_s)

% ＜TEST＞compare real entropy and surrogate
% The smaller the entropy value, the less dispersed
if H_sum < entropy_s_inf95
    disp('SIGNIFICANT!!!');
else
    disp('Not significant..');
end

surrogate_p=entropy_s_sort;

Pv=(([1:1e4]/1e4)');
surrogate_p(:,2)=Pv;

[IDX]=knnsearch(surrogate_p(:,1),H_sum);
ao=round(surrogate_p(IDX),4);
fao=find(round(surrogate_p(:,1),4)==ao);

mfao=max(fao);
pvalue=Pv(mfao);

%% MDS(fig5)
% 210623
% 210716 update (after cross-validation)
% load x_test from real_cv
% import data(x_test)
csvfiles=dir('*.csv');%be careful for the PATH!!
numfiles=length(csvfiles);
x_test=cell(1,numfiles);

for k=1:numfiles
    x_test{k}=importdata(csvfiles(k).name);
end
clear k numfiles

x_test = x_test';
x_test = cell2mat(x_test);

decoded_vm = decoded_vm';
decoded_vm = cell2mat(decoded_vm);

% if there is duplicate in x_test, MDS cant be executed. → delete one in that case
for j= 1:size(x_test,1)
    for i=1:size(x_test,1)
        F(i,j) = isequal(x_test(j,:) , x_test(i,:));
    end
end
clear i j

FF = triu(F,1);
FFsum = sum(FF,2);
index_exclude = find(FFsum == 1); %duplicated idx

x_test(index_exclude,:)=[];
decoded_vm(index_exclude,:)=[];

% creating input arguments required for MDS execution（every RMSE in each SW）
for i=1:size(x_test,1)
    for j=1:size(x_test,1)
        RMSE_xtest(i,j) = sqrt(immse(x_test(i,:),x_test(j,:)));
    end
end
clear i j

Y = mdscale(RMSE_xtest,2,'criterion','metricstress');

% calculate RMSE_judge using RMSE_judge.m
figure;
subplot(2,2,1);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,1),color);title('cell #1');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,2);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,2),color);title('cell #2');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,3);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,3),color);title('cell #3');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,4);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,4),color);title('cell #4');xlabel('MDS1');ylabel('MDS2');


%% correlation coefficient of the RMSE scores between MC pairs 
clear v C
v=1:3; C=nchoosek(v,2); % v=1:4; v=1:5;

for i=1:size(C,1)
    rr{i}=corrcoef(RMSE_judge_3c1(:,C(i,1)),RMSE_judge_3c1(:,C(i,2)));% RMSE_judge_4c1,RMSE_judge_5c1
    r(i,:)=rr{1,i}(1,2);
end
clear i
r_all = [r_all;r]; %r_all_fix...from 3celldata ⇒ 5celldata
clearvars -except r_all

%% overlap
cell1 = find(RMSE_judge_5c1(:,1)==1);
cell2 = find(RMSE_judge_5c1(:,2)==1);
cell3 = find(RMSE_judge_5c1(:,3)==1);
cell4 = find(RMSE_judge_5c1(:,4)==1);
cell5 = find(RMSE_judge_5c1(:,5)==1);
whole = [1:1:size(RMSE_judge_5c1,1)];

Cell={cell1 cell2 cell3 cell4 cell5};
v=1:1:5; C=nchoosek(v,2); % v=1:1:3 v=1:1:4
clear i
for i=1:size(C,1)
    mem = ismember(Cell{1,C(i,1)},Cell{1,C(i,2)});
    memsum = sum(mem);
    overlap(i,:) = memsum / numel(whole)*100;
end
save('overlap','overlap')
overlap_all = [overlap_all;overlap];

figure;
histogram(overlap_all);xlabel('Overlap % per total');ylabel('Count')

%% relationship of SW features and predictable SW

% 220929
% 221230 Update

load('SW.mat', 'RMSE_judge_3c1')
% load('SW.mat', 'spwpeakampraw')
% load('SW.mat', 'spwdurraw')
% load('SW.mat', 'AUCsum')
% load('SW.mat', 'Pmaxrgpower')
load('SW.mat', 'PmaxrgHz')

RMSE_judge_sum = sum(RMSE_judge_3c1,2);

pred0 = find(RMSE_judge_sum==0);
pred1 = find(RMSE_judge_sum==1);
pred2 = find(RMSE_judge_sum==2);
pred3 = find(RMSE_judge_sum==3);
pred4 = find(RMSE_judge_sum==4);

% amplitude
% sw0_amp = spwpeakampraw(pred0);
% sw1_amp = spwpeakampraw(pred1);
% sw2_amp = spwpeakampraw(pred2);
% sw3_amp = spwpeakampraw(pred3);
% sw4_amp = spwpeakampraw(pred4);
% 
% SW_amp = [sw0_amp;sw1_amp;sw2_amp;sw3_amp;sw4_amp];

% duration
% sw0_dur = spwdurraw(pred0);
% sw1_dur = spwdurraw(pred1);
% sw2_dur = spwdurraw(pred2);
% sw3_dur = spwdurraw(pred3);
% sw4_dur = spwdurraw(pred4);
% 
% SW_dur = [sw0_dur;sw1_dur;sw2_dur;sw3_dur;sw4_dur];

% FFT
% for i=1:size(RMSE_judge_4c1,1)
%     [Pmaxrgpower(i,:),PmaxrgHz(i,:),AUCsum(i,:)]=fft_ao2(peak50ms_n(i,:),fs,t);
% end
% clear i

% % max ripple Hz
sw0_Hz = PmaxrgHz(pred0);
sw1_Hz = PmaxrgHz(pred1);
sw2_Hz = PmaxrgHz(pred2);
sw3_Hz = PmaxrgHz(pred3);
sw4_Hz = PmaxrgHz(pred4);

SW_Hz = [sw0_Hz;sw1_Hz;sw2_Hz;sw3_Hz;sw4_Hz];


% ripple AUC(120-250Hz)
% sw0_auc = AUCsum(pred0);
% sw1_auc = AUCsum(pred1);
% sw2_auc = AUCsum(pred2);
% sw3_auc = AUCsum(pred3);
% sw4_auc = AUCsum(pred4);
% 
% SW_auc = [sw0_auc;sw1_auc;sw2_auc;sw3_auc;sw4_auc];

% max ripple power(in range of 120-250Hz)
% sw0_pw = Pmaxrgpower(pred0);
% sw1_pw = Pmaxrgpower(pred1);
% sw2_pw = Pmaxrgpower(pred2);
% sw3_pw = Pmaxrgpower(pred3);
% sw4_pw = Pmaxrgpower(pred4);
% 
% SW_pw = [sw0_pw;sw1_pw;sw2_pw;sw3_pw;sw4_pw];

% 
% P = [repmat(0,numel(pred0),1);...
%     repmat(1,numel(pred1),1);...
%     repmat(2,numel(pred2),1);...
%     repmat(3,numel(pred3),1);...
%     repmat(4,numel(pred4),1)];

% figure;
% scatter(P+1,SW_Hz);xlim([0 4]);hold on;
% boxplot(SW_Hz,P);
% ylabel('frequency (Hz)');

% pred4 = find(RMSE_judge_sum==4);
% sw4_amp = spwpeakampraw(pred4);
% sw4_dur = spwdurraw(pred4);
% sw4_auc = AUCsum(pred4);
% sw4_pw = Pmaxrgpower(pred4);

% amplitude
% SW_amp_m = [mean(sw0_amp);mean(sw1_amp);mean(sw2_amp);mean(sw3_amp);mean(sw4_amp)];

% duration
% SW_dur_m = [mean(sw0_dur);mean(sw1_dur);mean(sw2_dur);mean(sw3_dur);mean(sw4_dur)];

% % max ripple Hz
SW_Hz_m = [mean(sw0_Hz);mean(sw1_Hz);mean(sw2_Hz);mean(sw3_Hz);mean(sw4_Hz)];

% ripple AUC(120-250Hz)
% SW_auc_m = [mean(sw0_auc);mean(sw1_auc);mean(sw2_auc);mean(sw3_auc);mean(sw4_auc)];

% max ripple power(in range of 120-250Hz)
% SW_pw_m = [mean(sw0_pw);mean(sw1_pw);mean(sw2_pw);mean(sw3_pw);mean(sw4_pw)];

% A=isnan(SW_amp_m);
% SW_amp_m_z = SW_amp_m;
% SW_amp_m_z(A)=[];clear A
% SW_amp_m_z = zscore(SW_amp_m_z);
% 
% A=isnan(SW_dur_m);
% SW_dur_m_z = SW_dur_m;
% SW_dur_m_z(A)=[];clear A
% SW_dur_m_z = zscore(SW_dur_m_z);
% 
% A=isnan(SW_auc_m);
% SW_auc_m_z = SW_auc_m;
% SW_auc_m_z(A)=[];clear A
% SW_auc_m_z = zscore(SW_auc_m_z);
% 
% A=isnan(SW_pw_m);
% SW_pw_m_z = SW_pw_m;
% SW_pw_m_z(A)=[];clear A
% SW_pw_m_z = zscore(SW_pw_m_z);


A=isnan(SW_Hz_m);
SW_Hz_m_z = SW_Hz_m;
SW_Hz_m_z(A)=[];clear A
SW_Hz_m_z = zscore(SW_Hz_m_z);
% 
% SW_amp_m_z_all = [SW_amp_m_z];
% SW_dur_m_z_all = [SW_dur_m_z];
% SW_auc_m_z_all = [SW_auc_m_z];
% SW_pw_m_z_all = [SW_pw_m_z];
% SW_Hz_m_z_all = [SW_Hz_m_z];

% SW_amp_m_z_all = [SW_amp_m_z_all {SW_amp_m_z}];
% SW_dur_m_z_all = [SW_dur_m_z_all {SW_dur_m_z}];
% SW_auc_m_z_all = [SW_auc_m_z_all {SW_auc_m_z}];
% SW_pw_m_z_all = [SW_pw_m_z_all {SW_pw_m_z}];
SW_Hz_m_z_all = [SW_Hz_m_z_all {SW_Hz_m_z}];


% SW_amp_m_all=[SW_amp_m];
% SW_dur_m_all=[SW_dur_m];
% SW_auc_m_all=[SW_auc_m];
% SW_pw_m_all=[SW_pw_m];
% SW_Hz_m_all=[SW_Hz_m];

% SW_amp_m_all=[SW_amp_m_all {SW_amp_m}];
% SW_dur_m_all=[SW_dur_m_all {SW_dur_m}];
SW_Hz_m_all=[SW_Hz_m_all {SW_Hz_m}];
% SW_auc_m_all=[SW_auc_m_all {SW_auc_m}];
% SW_pw_m_all=[SW_pw_m_all {SW_pw_m}];

for i=1:size(SW_amp_m_all,2)
    val_idx{i} = (find(SW_amp_m_all{1,i}>0)); % Zスコア化するとマイナスの数値も入るので注意
end
clear i


% #pred 0 (1)
for i=1:size(SW_amp_m_z_all,2)
    SW_Hz_0(i) = SW_Hz_m_z_all{1,i}(1);
end

% #pred 1 (2)
for i=1:size(SW_amp_m_z_all,2)
    SW_Hz_1(i) = SW_Hz_m_z_all{1,i}(2);
end

% #pred 2 (3)
for i=1:size(SW_amp_m_z_all,2)
    A_pred2(i,:) = sum(ismember(val_idx{1,i},3));% 3 があるidxを抽出
end
A_pred2_idx = find(A_pred2 == 1)';
for j=1:size(A_pred2_idx,2)
    SW_Hz_2(j)=SW_Hz_m_z_all{1,A_pred2_idx(j)}(3);
end

% #pred 3 (4) #3番目だけ4番目に4がない
for i=1:size(SW_amp_m_z_all,2)
    A_pred3(i,:) = sum(ismember(val_idx{1,i},4));
end
A_pred3_idx = find(A_pred3 == 1)';%ひとまず3番目を除く
A_pred3_idx(2)=[];%A_pred3_idxでは2番目
for j=1:size(A_pred3_idx,2)
    SW_pw_3(j)=SW_pw_m_z_all{1,A_pred3_idx(j)}(4);
end
SW_pw_3 = [SW_pw_3 SW_pw_m_z_all{1,3}(3)];

% #pred 4 (5) 
for i=1:size(SW_amp_m_z_all,2)
    A_pred4(i,:) = sum(ismember(val_idx{1,i},5));
end
A_pred4_idx = find(A_pred4 == 1)';
for j=1:size(A_pred4_idx,2)
    SW_Hz_4(j)=SW_Hz_m_z_all{1,A_pred4_idx(j)}(5);
end


P = [repmat(0,25,1);...
repmat(1,25,1);...
repmat(2,24,1);...
repmat(3,10,1);
repmat(4,4,1)];

SW_amp = [SW_amp_0 SW_amp_1 SW_amp_2 SW_amp_3 SW_amp_4];
SW_dur = [SW_dur_0 SW_dur_1 SW_dur_2 SW_dur_3 SW_dur_4];
SW_auc = [SW_auc_0 SW_auc_1 SW_auc_2 SW_auc_3 SW_auc_4];
SW_pw = [SW_pw_0 SW_pw_1 SW_pw_2 SW_pw_3 SW_pw_4];
SW_Hz = [SW_Hz_0 SW_Hz_1 SW_Hz_2 SW_Hz_3 SW_Hz_4];

% Fig.6E
figure;
swarmchart(P,SW_amp,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_amp,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('SW amplitude (Z-scored)')
% regression line
Y = [median(SW_amp(P==0)) median(SW_amp(P==1)) median(SW_amp(P==2)) median(SW_amp(P==3)) median(SW_amp(P==4))]';
X = [1:5]';
s = X\Y; % regression coefficient
yCalc1 = s*X;
figure;
plot(X-1,yCalc1,'Color',[0.74 0.74 0.74]);hold on;
swarmchart(P,SW_amp,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_amp,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('SW amplitude (Z-scored)')

% Fig.S6
figure;
subplot(1,4,1)
swarmchart(P,SW_dur,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_dur,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('SW duration (Z-scored)')
subplot(1,4,2)
swarmchart(P,SW_auc,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_auc,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('AUC (Z-scored)')
subplot(1,4,3)
swarmchart(P,SW_pw,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_pw,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('FFT power (Z-scored)')
subplot(1,4,4)
swarmchart(P,SW_Hz,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_Hz,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('Ripple frequency (Z-scored)')


%% wavelet coherence
% 220513

% sample
% [wcoh,wcs,f] = wcoherence(x_test(1,:),m_LFP_s(1,:),20000); 20000 = sampling frq
% f; y-axis of the pseudo color plot output
% f;index 60~72 => 120-250 Hz, extract the index of wcoh
% range of 20～80ms is appropriate?(t index =>400-1600)→ vivo; 40~160ms(T(1*4001)index,800-3200)


fs = 20000;

for i = 1:size(x_test,1)
    [wcoh] = wcoherence(x_test(i,:),m_LFP_s(i,:),fs);
    w(i,:) = mean(wcoh(60:72,800:3200));% vitro; 400:1600、vivo; 800:3200
end
clear i
% m = mean(w,2);
% % mean(m)
% % % % 
% SWnum = size(w,1)/combset;
% W=[];
%  for i = 1 : combset
%     W(i,:) = mean(w(1+(i-1)*SWnum:SWnum*i,:));
%  end
%  mean(W)
% % % figure;plot(t,M,'k');hold on; plot(t,m2,'r');

