%% DNN_MC2.m
%% Update: 220209

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
% 5-fold cv、100-times shuffle each in python（total 50 of x_test and decoded_vm_s）
% clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
% clear LFP_sort LFP_sort_10_20 LFP_sort_mean combset shufflenum

combset = 10; % number of combinations
% shufflenum = 10;

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
% 
% for j = 1:size(decoded_vm,2)
%     for i = 1:size(x_test{1,1},1)
%         dist{:,j}(i,:)=dtw(x_test{:,j}(i,:),m_LFP_s{:,j}(i,:));
%     end
% end
% clear i
% 
% dist = cell2mat(dist);
% dist = reshape(dist,[size(base_corr_comb,1)*5,combset]);
% % 
% % dist = cell2mat(dist);
% % RMSE_cv = reshape(dist,[size(base_corr_comb,1)*25,1]);% RMSE_cv contains X x Y matrix [X; SW number, Y; shufflenum]
% % 
% % % RMSE（original SW trace used as training data vs corrected decoded SW waveform）
% for j=1:size(decoded_vm,2)
%     for i=1:size(x_test{1,1},1)
%         RMSE_cv{:,j}(i,:) = sqrt(immse(x_test{:,j}(i,:),m_LFP_s{:,j}(i,:)));
%     end
% end
% clear i j
% % 
% % % for j=1:size(decoded_vm,2)
% %     for i=1:size(x_test,1)
% %         RMSE_cv(i,:) = sqrt(immse(x_test(i,:),m_LFP_s(i,:)));
% %     end
% % % end
% % clear i j
% 
% RMSE_cv = cell2mat(RMSE_cv);
% RMSE_cv = reshape(RMSE_cv,[size(base_corr_comb,1)*5,combset]);% RMSE_cv contains X x Y matrix [X; SW number, Y; shufflenum]

% RMSE_cv_m_c5 = mean(RMSE_cv,2);

% clearvars -except RMSE_cv_m_c1 RMSE_cv_m_c2 RMSE_cv_m_c3 RMSE_cv_m_c4 RMSE_cv_m_c5
%%
t=(1:1:2001)/20000;
x_test = x_test';
m_LFP_s = m_LFP_s';
x_test = cell2mat(x_test);
m_LFP_s = cell2mat(m_LFP_s);

for i=1:size(x_test,1)
    RMSE_cv(i,:) = sqrt(immse(x_test(i,:),m_LFP_s(i,:)));
end
% RMSE_cv = reshape(RMSE_cv,[178,5]);

figure;
for i=1:size(x_test,1)% x_test_org; excluded duplicated SW
    subplot(14,14,i); hold on;% change value in each dataset
    plot(t,x_test(i,:),'k');xlim([0 0.1]);hold on;
    plot(t,m_LFP_s(i,:),'r');
end

IE=[];
for i=1:combset
    IE(i,:)=[143+180*(i-1),144+180*(i-1)];
end
IE= reshape(IE,[combset*2,1]);
x_test(IE,:)=[];
m_LFP_s(IE,:)=[];

%% wavelet coherence
% 220513

% sample
% [wcoh,wcs,f] = wcoherence(x_test(1,:),m_LFP_s(1,:),20000); %
% 20000はsampling frq
% f はdefaultで出力される擬似カラープロットのy軸
% fのindex 60~75が,100-250 Hzに該当するのでwcohのそのindexを抽出する
% 円錐状影響圏を考慮すると、0～100 msのウェーブレットコヒーレンスのうち、20～80msの範囲が妥当？(tのindexで400-1600)

fs = 20000;

for i = 1:size(x_test,1)
    [wcoh] = wcoherence(x_test(i,:),m_LFP_s(i,:),fs);
    w(i,:) = mean(wcoh(60:75,400:1600));
end
m = mean(w);

% figure;plot(t,M,'k');hold on; plot(t,m2,'r');



%% cumulative probability (fig.3c)

[f,x] = ecdf(RMSE_cv_m_c1);
[g,y] = ecdf(RMSE_cv_m_c2);
[h,z] = ecdf(RMSE_cv_m_c3);
[i,a] = ecdf(RMSE_cv_m_c4);
[j,b] = ecdf(RMSE_cv_m_c5);


% [H,P,K] = kstest2(x,X)

figure;plot(x,f);hold on;plot(y,g);hold on;plot(z,h);hold on;plot(a,i);hold on;plot(b,j);
ylabel('cumulative probability');xlabel('RMSE');%title('all slices');xlim([0 0.25]);
legend('cell#1','cell#2','cell#3','cell#4','cell#5','Location','southeast');

%% MDS(fig4A)
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

figure;scatter(Y(:,1),Y(:,2),'.')

% MDS領域を任意で5つに振り分ける
idx = kmeans(Y,5);
figure;gscatter(Y(:,1),Y(:,2),idx);%確認用

idx1=find(idx==1);
idx2=find(idx==2);
idx3=find(idx==3);
idx4=find(idx==4);
idx5=find(idx==5);

% Cell1,各MDS領域におけるRMSE
RMSE_cv_m_c1_idx1=RMSE_cv_m_c1(idx1);
RMSE_cv_m_c1_idx2=RMSE_cv_m_c1(idx2);
RMSE_cv_m_c1_idx3=RMSE_cv_m_c1(idx3);
RMSE_cv_m_c1_idx4=RMSE_cv_m_c1(idx4);
RMSE_cv_m_c1_idx5=RMSE_cv_m_c1(idx5);

% Cell2
RMSE_cv_m_c2_idx1=RMSE_cv_m_c2(idx1);
RMSE_cv_m_c2_idx2=RMSE_cv_m_c2(idx2);
RMSE_cv_m_c2_idx3=RMSE_cv_m_c2(idx3);
RMSE_cv_m_c2_idx4=RMSE_cv_m_c2(idx4);
RMSE_cv_m_c2_idx5=RMSE_cv_m_c2(idx5);

%Cell3
RMSE_cv_m_c3_idx1=RMSE_cv_m_c3(idx1);
RMSE_cv_m_c3_idx2=RMSE_cv_m_c3(idx2);
RMSE_cv_m_c3_idx3=RMSE_cv_m_c3(idx3);
RMSE_cv_m_c3_idx4=RMSE_cv_m_c3(idx4);
RMSE_cv_m_c3_idx5=RMSE_cv_m_c3(idx5);

% Cell4
RMSE_cv_m_c4_idx1=RMSE_cv_m_c4(idx1);
RMSE_cv_m_c4_idx2=RMSE_cv_m_c4(idx2);
RMSE_cv_m_c4_idx3=RMSE_cv_m_c4(idx3);
RMSE_cv_m_c4_idx4=RMSE_cv_m_c4(idx4);
RMSE_cv_m_c4_idx5=RMSE_cv_m_c4(idx5);

% Cell5
RMSE_cv_m_c5_idx1=RMSE_cv_m_c5(idx1);
RMSE_cv_m_c5_idx2=RMSE_cv_m_c5(idx2);
RMSE_cv_m_c5_idx3=RMSE_cv_m_c5(idx3);
RMSE_cv_m_c5_idx4=RMSE_cv_m_c5(idx4);
RMSE_cv_m_c5_idx5=RMSE_cv_m_c5(idx5);


clear f x g y j z i a j b
[f,x] = ecdf(RMSE_cv_m_c1_idx5);
[g,y] = ecdf(RMSE_cv_m_c2_idx5);
[h,z] = ecdf(RMSE_cv_m_c3_idx5);
[i,a] = ecdf(RMSE_cv_m_c4_idx5);
[j,b] = ecdf(RMSE_cv_m_c5_idx5);


% [h,p,k] = kstest2(y,z)

figure;plot(x,f);hold on;plot(y,g);hold on;plot(z,h);hold on;plot(a,i);hold on;plot(b,j);
ylabel('cumulative probability');xlabel('RMSE');title('MDS #1 area');%xlim([0 0.25]);
legend('cell#1','cell#2','cell#3','cell#4','cell#5','Location','southeast');

%% ## correct baseline & calculate RMSE ##
% Update 220313
% バンドパスした波形（100-250Hz）同士を比較したとき、RMSEが細胞数を増やすと小さくなるか？？
% ## k-fold cross-validation ##
% 5-fold cv、100-times shuffle each in python（total 50 of x_test and decoded_vm_s）
% clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
% clear LFP_sort LFP_sort_10_20 LFP_sort_mean combset shufflenum


% Average of the bottom 10-20% of LFP source data（①）
for i = 1 : size(x_test_filt,1)
    LFP_sort(i,:) = sort(x_test_filt(i,:));
    LFP_sort_10_20(i,:) = LFP_sort(i,200:400);
    LFP_sort_mean(i,:) = mean(LFP_sort_10_20(i,:),2);
end
clear i 

% Average of the bottom 10-20% of decoded SW waveform（②）
for i = 1 : size(decoded_vm_filt,1)
    vmsort(i,:) = sort(decoded_vm_filt(i,:));
    vmsort_10_20(i,:) = vmsort(i,200:400);
    vmsort_mean(i,:)= mean(vmsort_10_20(i,:),2);
end
clear i

% correct baseline（② - ①）
% vmsort_double = cell2mat(vmsort_mean);
% LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_mean - LFP_sort_mean;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(decoded_vm_filt,1)
    m_LFP(i,:) = decoded_vm_filt(i,:) - base_corr_comb(i,:);
end
clear i

% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for i=1:size(decoded_vm_filt,1)
    RMSE_5cell(i,:) = sqrt(immse(x_test_filt(i,:),m_LFP_s(i,:)));
end
clear i

% RMSE_cv = cell2mat(RMSE_cv);
% RMSE_cv = reshape(RMSE_cv,[size(base_corr_comb,1)*5,1]);% RMSE_cv contains X x Y matrix [X; SW number, Y; shufflenum]
% 
% RMSE_cv_m_c5 = mean(RMSE_cv,2);

% clearvars -except RMSE_cv_m_c1 RMSE_cv_m_c2 RMSE_cv_m_c3 RMSE_cv_m_c4 RMSE_cv_m_c5


%% ## correct baseline & calculate RMSE ##
% Update 220313
% バンドパスした波形（100-250Hz）同士を比較したとき、RMSEが細胞数を増やすと小さくなるか？？
% ## k-fold cross-validation ##
% 5-fold cv、100-times shuffle each in python（total 50 of x_test and decoded_vm_s）
% clear vmsort vmsort_10_20 vmsort_mean csvfiles numfiles vmsort_double base_corr_comb m_LFP_s LFPsort_double
% clear LFP_sort LFP_sort_10_20 LFP_sort_mean combset shufflenum


% Average of the bottom 10-20% of LFP source data（①）
for i = 1 : size(x_test_filt,1)
    LFP_sort(i,:) = sort(x_test_filt(i,:));
    LFP_sort_10_20(i,:) = LFP_sort(i,200:400);
    LFP_sort_mean(i,:) = mean(LFP_sort_10_20(i,:),2);
end
clear i 

% Average of the bottom 10-20% of decoded SW waveform（②）
for i = 1 : size(decoded_vm_1_2_filt,1)
    vmsort(i,:) = sort(decoded_vm_1_2_filt(i,:));
    vmsort_10_20(i,:) = vmsort(i,200:400);
    vmsort_mean(i,:)= mean(vmsort_10_20(i,:),2);
end
clear i

% correct baseline（② - ①）
% vmsort_double = cell2mat(vmsort_mean);
% LFPsort_double = cell2mat(LFP_sort_mean);
base_corr_comb = vmsort_mean - LFP_sort_mean;

% substract the corrected baseline from the decoded SW waveform
for i = 1 : size(decoded_vm_1_2_filt,1)
    m_LFP_s(i,:) = decoded_vm_1_2_filt(i,:) - base_corr_comb(i,:);
end
clear i

% RMSE（original SW trace used as training data vs corrected decoded SW waveform）
for i=1:size(decoded_vm_filt,1)
    RMSE_1cell(i,:) = sqrt(immse(x_test_filt(i,:),m_LFP_s(i,:)));
end
clear i

% RMSE_cv = cell2mat(RMSE_cv);
% RMSE_cv = reshape(RMSE_cv,[size(base_corr_comb,1)*5,1]);% RMSE_cv contains X x Y matrix [X; SW number, Y; shufflenum]
% 
% RMSE_cv_m_c5 = mean(RMSE_cv,2);

% clearvars -except RMSE_cv_m_c1 RMSE_cv_m_c2 RMSE_cv_m_c3 RMSE_cv_m_c4 RMSE_cv_m_c5
