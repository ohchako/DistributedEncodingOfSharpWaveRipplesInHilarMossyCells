%% RMSE_judge.m
% 210615
% update 240310

RMSE_corr_cv_3c1_s1=RMSE_corr_cv_s(:,1:100);
RMSE_corr_cv_3c1_s2=RMSE_corr_cv_s(:,101:200);
RMSE_corr_cv_3c1_s3=RMSE_corr_cv_s(:,201:300);

RMSE_corr_cv_3c1_s1_sort=sort(RMSE_corr_cv_3c1_s1,2);
RMSE_corr_cv_3c1_s2_sort=sort(RMSE_corr_cv_3c1_s2,2);
RMSE_corr_cv_3c1_s3_sort=sort(RMSE_corr_cv_3c1_s3,2);

RMSE_corr_3c1_s_sort={RMSE_corr_cv_3c1_s1_sort RMSE_corr_cv_3c1_s2_sort RMSE_corr_cv_3c1_s3_sort};

% 1 if SW is significantly predictable (significantly lower RMSE) compared to shuffle, 0 otherwise
RMSE_judge_3c1 = [];
for i = 1 : size(RMSE_corr_3c1_s_sort,2)
    for j = 1 : size(RMSE_corr_cv,1)
        if RMSE_corr_cv(j,i) < RMSE_corr_3c1_s_sort{:,i}(j,5)
            RMSE_judge_3c1(j,i)=1;
        else
            RMSE_judge_3c1(j,i)=0;
        end
    end
end
clear i j


% real rate
% 3c1
for i=1:3
    j_rate(:,i) = numel(find(RMSE_judge_3c1(:,i)==1))/size(RMSE_judge_3c1,1)*100;
end

% 3c2
v = 1:1:3; C = nchoosek(v,2);
for i = 1:size(C,1)
    A = RMSE_judge_3c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_3c2(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_3c1,1)*100;
    clear A Asum
end
clear i

% 3c3
clear C
C = nchoosek(v,3);
A = RMSE_judge_3c1(:,C(1,:));
Asum = sum(A,2);
j_rate_3c3 = numel(find(Asum > 0)) / size(RMSE_judge_3c1,1)*100;

% KS test
% RMSE_shuffle = {reshape(RMSE_corr_cv_3c1_s1,[],1) reshape(RMSE_corr_cv_3c1_s2,[],1) reshape(RMSE_corr_cv_3c1_s3,[],1)};
% 
% clear f x F X h p k
% for i=1:size(RMSE_corr_cv_3c1,2)
%     [f(:,i),x(:,i)] = ecdf(RMSE_corr_cv_3c1(:,i));
%     [F(:,i),X(:,i)] = ecdf(RMSE_shuffle{1,i});
%     [~,p(i,:),k(i,:)] = kstest2(x(:,i),X(:,i));
% end


% RMSE_judgeのshuffle
for i= 1:size(RMSE_judge_3c1,2)
    for j = 1:100
        R{:,j}(:,i) = randperm(size(RMSE_judge_3c1,1));
    end
end
clear i j 

for j = 1:100
    for i=1:size(RMSE_judge_3c1,2)
        A = RMSE_judge_3c1(:,i);
        RMSE_judge_s{:,j}(:,i) = A(R{1,j}(:,i));
        clear A
    end
end
clear i j

% shuffle rate
% 3c1
for j = 1:100
    for i=1:size(RMSE_judge_3c1,2)
        j_rate_s{:,j}(:,i) = numel(find(RMSE_judge_s{:,j}(:,i) == 1)) / size(RMSE_judge_3c1,1)*100;
    end
end
clear i j
j_rate_s = cell2mat(j_rate_s);
j_rate_s_sort = sort(j_rate_s);
j_rate_s_95CI_l = j_rate_s_sort(:,size(j_rate_s_sort,2)*0.05);
j_rate_s_95CI_h = j_rate_s_sort(:,size(j_rate_s_sort,2)-size(j_rate_s_sort,2)*0.05);

% 3c2
clear C
v = 1:1:3; C = nchoosek(v,2);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_3c2_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_3c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_3c2_s = cell2mat(j_rate_3c2_s);
j_rate_3c2_s_sort = sort(j_rate_3c2_s);
j_rate_3c2_95CI_l = j_rate_3c2_s_sort(:,size(j_rate_3c2_s_sort,2)*0.05);
j_rate_3c2_95CI_h = j_rate_3c2_s_sort(:,size(j_rate_3c2_s_sort,2)-size(j_rate_3c2_s_sort,2)*0.05);

% 3c3
clear C
C = nchoosek(v,3);
for i = 1:100
    A = RMSE_judge_s{:,i}(:,C(1,:));
    Asum = sum(A,2);
    j_rate_3c3_s{:,i} = numel(find(Asum > 0)) / size(RMSE_judge_3c1,1)*100;
    clear A Asum
end
clear i 
j_rate_3c3_s = cell2mat(j_rate_3c3_s);
j_rate_3c3_s_sort = sort(j_rate_3c3_s);
j_rate_3c3_95CI_l = j_rate_3c3_s_sort(:,size(j_rate_3c3_s_sort,2)*0.05);
j_rate_3c3_95CI_h = j_rate_3c3_s_sort(:,size(j_rate_3c3_s_sort,2)-size(j_rate_3c3_s_sort,2)*0.05);



CI95_h = [j_rate_s_95CI_h j_rate_4c2_95CI_h j_rate_4c3_95CI_h j_rate_4c4_95CI_h]';
CI95_l = [j_rate_s_95CI_l j_rate_4c2_95CI_l j_rate_4c3_95CI_l j_rate_4c4_95CI_l]';


j_rate_all = [{j_rate'} {j_rate_3c2'} {j_rate_3c3'}];
J_rate_all = [J_rate_all;j_rate_all];

CI95_h = [{j_rate_s_95CI_h} {j_rate_3c2_95CI_h} {j_rate_3c3_95CI_h}]';
CI95_l = [{j_rate_s_95CI_l} {j_rate_3c2_95CI_l} {j_rate_3c3_95CI_l}]';

CI95_H = [CI95_H CI95_h];
CI95_L = [CI95_L CI95_l];

clearvars -except CI95_H CI95_L J_rate_all CI95_H_5cells CI95_L_5cells J_rate_all_5cells CI95_H_4cells CI95_L_4cells J_rate_all_4cells

% for saving
CI95_H = [{j_rate_s_95CI_h} {j_rate_3c2_95CI_h} {j_rate_3c3_95CI_h}]';
CI95_L = [{j_rate_s_95CI_l} {j_rate_3c2_95CI_l} {j_rate_3c3_95CI_l}]';


%% 4cells!!!
% 210615
% Update 220920

RMSE_corr_4c1_s1=RMSE_corr_cv_s(:,1:100);
RMSE_corr_4c1_s2=RMSE_corr_cv_s(:,101:200);
RMSE_corr_4c1_s3=RMSE_corr_cv_s(:,201:300);
RMSE_corr_4c1_s4=RMSE_corr_cv_s(:,301:400);

RMSE_corr_4c1_s1_sort=sort(RMSE_corr_4c1_s1,2);
RMSE_corr_4c1_s2_sort=sort(RMSE_corr_4c1_s2,2);
RMSE_corr_4c1_s3_sort=sort(RMSE_corr_4c1_s3,2);
RMSE_corr_4c1_s4_sort=sort(RMSE_corr_4c1_s4,2);

RMSE_corr_4c1_s_sort={RMSE_corr_4c1_s1_sort RMSE_corr_4c1_s2_sort RMSE_corr_4c1_s3_sort RMSE_corr_4c1_s4_sort};

RMSE_judge_4c1 = [];
for i = 1 : size(RMSE_corr_4c1_s_sort,2)
    for j = 1 : size(RMSE_corr_cv,1)
        if RMSE_corr_cv(j,i) < RMSE_corr_4c1_s_sort{:,i}(j,5)
            RMSE_judge_4c1(j,i)=1;
        else
            RMSE_judge_4c1(j,i)=0;
        end
    end
end

% real rate
% 4c1
for i=1:4
    j_rate(:,i) = numel(find(RMSE_judge_4c1(:,i)==1))/size(RMSE_judge_4c1,1)*100;
end

% 4c2
v = 1:1:4; C = nchoosek(v,2);
for i = 1:size(C,1)
    A = RMSE_judge_4c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_4c2(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;
    clear A Asum
end
clear i

% 4c3
clear C
C = nchoosek(v,3);
for i = 1:size(C,1)
    A = RMSE_judge_4c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_4c3(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;
    clear A Asum
end
clear i

% 4c4
clear C
C = nchoosek(v,4);
A = RMSE_judge_4c1(:,C(1,:));
Asum = sum(A,2);
j_rate_4c4 = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;

% % KS test
% RMSE_shuffle = {reshape(RMSE_corr_4c1_s1,[],1) reshape(RMSE_corr_4c1_s2,[],1) reshape(RMSE_corr_4c1_s3,[],1) reshape(RMSE_corr_4c1_s4,[],1)};
% 
% clear f x F X h p k
% for i=1:size(RMSE_corr_cv,2)
%     [f(:,i),x(:,i)] = ecdf(RMSE_corr_cv(:,i));
%     [F(:,i),X(:,i)] = ecdf(RMSE_shuffle{1,i});
%     [~,p(i,:),k(i,:)] = kstest2(x(:,i),X(:,i));
% end

% 
% figure;
% subplot(1,4,1)
% imagesc(RMSE_judge_4c1(:,1));colormap(gray);set(gca,'YDir','normal');xticks([])
% title(['Prediction Rate: ',num2str(j_rate(:,1)),'%; Pvalue: ',num2str(p(1)),]); hold on;
% subplot(1,4,2)
% imagesc(RMSE_judge_4c1(:,2));colormap(gray);set(gca,'YDir','normal');xticks([])
% title(['Prediction Rate: ',num2str(j_rate(:,2)),'%; Pvalue: ',num2str(p(2)),]); hold on;
% subplot(1,4,3)
% imagesc(RMSE_judge_4c1(:,3));colormap(gray);set(gca,'YDir','normal');xticks([])
% title(['Prediction Rate: ',num2str(j_rate(:,3)),'%; Pvalue: ',num2str(p(3)),]); hold on;
% subplot(1,4,4)
% imagesc(RMSE_judge_4c1(:,4));colormap(gray);set(gca,'YDir','normal');xticks([])
% title(['Prediction Rate: ',num2str(j_rate(:,4)),'%; Pvalue: ',num2str(p(4)),]); hold on;
% 
% 
% cell1 = find(RMSE_judge_4c1(:,1)==1);
% cell2 = find(RMSE_judge_4c1(:,2)==1);
% cell3 = find(RMSE_judge_4c1(:,3)==1);
% cell4 = find(RMSE_judge_4c1(:,4)==1);
% 
% setListData = {cell1, cell2, cell3, cell4};
% setLabels = ['cell1'; 'cell2'; 'cell3'; 'cell4'];
% figure;
% h = vennEulerDiagram(setListData, setLabels);


% RMSE_judgeのshuffle

for i= 1:size(RMSE_judge_4c1,2)
    for j = 1:100
        R{:,j}(:,i) = randperm(size(RMSE_judge_4c1,1));
    end
end
clear i j 

for j = 1:100
    for i=1:size(RMSE_judge_4c1,2)
        A = RMSE_judge_4c1(:,i);
        RMSE_judge_s{:,j}(:,i) = A(R{1,j}(:,i));
        clear A
    end
end
clear i j

% shuffle rate
% 4c1
for j = 1:100
    for i=1:size(RMSE_judge_4c1,2)
        j_rate_s{:,j}(:,i) = numel(find(RMSE_judge_s{:,j}(:,i) == 1)) / size(RMSE_judge_4c1,1)*100;
    end
end
clear i j
j_rate_s = cell2mat(j_rate_s);
j_rate_s_sort = sort(j_rate_s);
j_rate_s_95CI_l = j_rate_s_sort(:,size(j_rate_s_sort,2)*0.05);
j_rate_s_95CI_h = j_rate_s_sort(:,size(j_rate_s_sort,2)-size(j_rate_s_sort,2)*0.05);

% 4c2
clear C
v = 1:1:4; C = nchoosek(v,2);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_4c2_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_4c2_s = cell2mat(j_rate_4c2_s);
j_rate_4c2_s_sort = sort(j_rate_4c2_s);
j_rate_4c2_95CI_l = j_rate_4c2_s_sort(:,size(j_rate_4c2_s_sort,2)*0.05);
j_rate_4c2_95CI_h = j_rate_4c2_s_sort(:,size(j_rate_4c2_s_sort,2)-size(j_rate_4c2_s_sort,2)*0.05);

% 4c3
clear C
v = 1:1:4; C = nchoosek(v,3);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_4c3_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_4c3_s = cell2mat(j_rate_4c3_s);
j_rate_4c3_s_sort = sort(j_rate_4c3_s);
j_rate_4c3_95CI_l = j_rate_4c3_s_sort(:,size(j_rate_4c3_s_sort,2)*0.05);
j_rate_4c3_95CI_h = j_rate_4c3_s_sort(:,size(j_rate_4c3_s_sort,2)-size(j_rate_4c3_s_sort,2)*0.05);

% 4c4
clear C
C = nchoosek(v,4);
for i = 1:100
    A = RMSE_judge_s{:,i}(:,C(1,:));
    Asum = sum(A,2);
    j_rate_4c4_s{:,i} = numel(find(Asum > 0)) / size(RMSE_judge_4c1,1)*100;
    clear A Asum
end
clear i 
j_rate_4c4_s = cell2mat(j_rate_4c4_s);
j_rate_4c4_s_sort = sort(j_rate_4c4_s);
j_rate_4c4_95CI_l = j_rate_4c4_s_sort(:,size(j_rate_4c4_s_sort,2)*0.05);
j_rate_4c4_95CI_h = j_rate_4c4_s_sort(:,size(j_rate_4c4_s_sort,2)-size(j_rate_4c4_s_sort,2)*0.05);



CI95_h = [j_rate_s_95CI_h j_rate_4c2_95CI_h j_rate_4c3_95CI_h j_rate_4c4_95CI_h]';
CI95_l = [j_rate_s_95CI_l j_rate_4c2_95CI_l j_rate_4c3_95CI_l j_rate_4c4_95CI_l]';


j_rate_all = [{j_rate'} {j_rate_4c2'} {j_rate_4c3'} {j_rate_4c4'}];
J_rate_all = [J_rate_all;j_rate_all];

CI95_h = [{j_rate_s_95CI_h} {j_rate_4c2_95CI_h} {j_rate_4c3_95CI_h} {j_rate_4c4_95CI_h}]';
CI95_l = [{j_rate_s_95CI_l} {j_rate_4c2_95CI_l} {j_rate_4c3_95CI_l} {j_rate_4c4_95CI_l}]';

CI95_H = [CI95_H CI95_h];
CI95_L = [CI95_L CI95_l];

clearvars -except CI95_H CI95_L J_rate_all CI95_H_5cells CI95_L_5cells J_rate_all_5cells

% 保存用（一番最初）
CI95_H = [{j_rate_s_95CI_h} {j_rate_4c2_95CI_h} {j_rate_4c3_95CI_h} {j_rate_4c4_95CI_h}]';
CI95_L = [{j_rate_s_95CI_l} {j_rate_4c2_95CI_l} {j_rate_4c3_95CI_l} {j_rate_4c4_95CI_l}]';

%% 5cells!!!
% 210616
% Updated 220916

RMSE_corr_5c1_s1=RMSE_corr_cv_s(:,1:100);
RMSE_corr_5c1_s2=RMSE_corr_cv_s(:,101:200);
RMSE_corr_5c1_s3=RMSE_corr_cv_s(:,201:300);
RMSE_corr_5c1_s4=RMSE_corr_cv_s(:,301:400);
RMSE_corr_5c1_s5=RMSE_corr_cv_s(:,401:500);

RMSE_corr_5c1_s1_sort=sort(RMSE_corr_5c1_s1,2);
RMSE_corr_5c1_s2_sort=sort(RMSE_corr_5c1_s2,2);
RMSE_corr_5c1_s3_sort=sort(RMSE_corr_5c1_s3,2);
RMSE_corr_5c1_s4_sort=sort(RMSE_corr_5c1_s4,2);
RMSE_corr_5c1_s5_sort=sort(RMSE_corr_5c1_s5,2);

RMSE_corr_5c1_s_sort={RMSE_corr_5c1_s1_sort RMSE_corr_5c1_s2_sort RMSE_corr_5c1_s3_sort RMSE_corr_5c1_s4_sort RMSE_corr_5c1_s5_sort};

RMSE_judge_5c1 = [];
for i = 1 : size(RMSE_corr_5c1_s_sort,2)
    for j = 1 : size(RMSE_corr_cv,1)
        if RMSE_corr_cv(j,i) < RMSE_corr_5c1_s_sort{:,i}(j,5)
            RMSE_judge_5c1(j,i)=1;
        else
            RMSE_judge_5c1(j,i)=0;
        end
    end
end

% real rate
%5c1
for i=1:5
    j_rate(:,i) = numel(find(RMSE_judge_5c1(:,i) == 1)) / size(RMSE_judge_5c1,1)*100;
end
clear i

%5c2
v = 1:1:5; C = nchoosek(v,2);
for i = 1:size(C,1)
    A = RMSE_judge_5c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_5c2(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
    clear A Asum
end
clear i

%5c3
clear C
C = nchoosek(v,3);
for i = 1:size(C,1)
    A = RMSE_judge_5c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_5c3(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
    clear A Asum
end
clear i

%5c4
clear C
C = nchoosek(v,4);
for i = 1:size(C,1)
    A = RMSE_judge_5c1(:,C(i,:));
    Asum = sum(A,2);
    j_rate_5c4(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
    clear A Asum
end
clear i

%5c5
clear C
C = nchoosek(v,5);
A = RMSE_judge_5c1(:,C(1,:));
Asum = sum(A,2);
j_rate_5c5 = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;

% % KS test
% RMSE_shuffle = {reshape(RMSE_corr_5c1_s1,[],1) reshape(RMSE_corr_5c1_s2,[],1) reshape(RMSE_corr_5c1_s3,[],1) reshape(RMSE_corr_5c1_s4,[],1) reshape(RMSE_corr_5c1_s5,[],1)};
% 
% clear f x F X h p k
% for i=1:size(RMSE_corr_cv,2)
%     [f(:,i),x(:,i)] = ecdf(RMSE_corr_cv(:,i));
%     [F(:,i),X(:,i)] = ecdf(RMSE_shuffle{1,i});
%     [~,p(i,:),k(i,:)] = kstest2(x(:,i),X(:,i));
% end
% 
% figure;
% subplot(1,5,1)
% imagesc(RMSE_judge_5c1(:,1));colormap(gray);set(gca,'YDir','normal');
% title(['Prediction Rate: ',num2str(j_rate(:,1)),'%; Pvalue: ',num2str(p(1)),]); hold on;
% subplot(1,5,2)
% imagesc(RMSE_judge_5c1(:,2));colormap(gray);set(gca,'YDir','normal');
% title(['Prediction Rate: ',num2str(j_rate(:,2)),'%; Pvalue: ',num2str(p(2)),]); hold on;
% subplot(1,5,3)
% imagesc(RMSE_judge_5c1(:,3));colormap(gray);set(gca,'YDir','normal');
% title(['Prediction Rate: ',num2str(j_rate(:,3)),'%; Pvalue: ',num2str(p(3)),]); hold on;
% subplot(1,5,4)
% imagesc(RMSE_judge_5c1(:,4));colormap(gray);set(gca,'YDir','normal');
% title(['Prediction Rate: ',num2str(j_rate(:,4)),'%; Pvalue: ',num2str(p(4)),]); hold on;
% subplot(1,5,5)
% imagesc(RMSE_judge_5c1(:,5));colormap(gray);set(gca,'YDir','normal');
% title(['Prediction Rate: ',num2str(j_rate(:,5)),'%; Pvalue: ',num2str(p(5)),]); hold on;
% 
% 
% plot vennEulerDiagram (file: "upload")
cell1 = find(RMSE_judge_5c1(:,1)==1);
cell2 = find(RMSE_judge_5c1(:,2)==1);
cell3 = find(RMSE_judge_5c1(:,3)==1);
cell4 = find(RMSE_judge_5c1(:,4)==1);
cell5 = find(RMSE_judge_5c1(:,5)==1);
whole = [1:1:size(RMSE_judge_5c1,1)];
setListData = {cell1, cell2, cell3, cell4, cell5, whole};
setLabels = ['cell1'; 'cell2'; 'cell3'; 'cell4'; 'cell5'; 'whole'];
figure;
h = vennEulerDiagram(setListData, setLabels);

% RMSE_judge shuffle

for i= 1:5
    for j = 1:100
        R{:,j}(:,i) = randperm(size(RMSE_judge_5c1,1));
    end
end
clear i j 

%　confirmation
% for i=1:5
%     A = RMSE_judge_5c1(:,i)
%     RMSE_judge_s(:,i) = A(R{1,1}(:,i));
%     clear A
% end
% clear i

for j = 1:100
    for i=1:5
        A = RMSE_judge_5c1(:,i);
        RMSE_judge_s{:,j}(:,i) = A(R{1,j}(:,i));
        clear A
    end
end
clear i j

% shuffle rate
%5c1
for j = 1:100
    for i=1:5
        j_rate_s{:,j}(:,i) = numel(find(RMSE_judge_s{:,j}(:,i) == 1)) / size(RMSE_judge_5c1,1)*100;
    end
end
clear i j
j_rate_s = cell2mat(j_rate_s);
j_rate_s_sort = sort(j_rate_s);
j_rate_s_95CI_l = j_rate_s_sort(:,size(j_rate_s_sort,2)*0.05);
j_rate_s_95CI_h = j_rate_s_sort(:,size(j_rate_s_sort,2)-size(j_rate_s_sort,2)*0.05);

%5c2
clear C
v = 1:1:5; C = nchoosek(v,2);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_5c2_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_5c2_s = cell2mat(j_rate_5c2_s);
j_rate_5c2_s_sort = sort(j_rate_5c2_s);
j_rate_5c2_95CI_l = j_rate_5c2_s_sort(:,size(j_rate_5c2_s_sort,2)*0.05);
j_rate_5c2_95CI_h = j_rate_5c2_s_sort(:,size(j_rate_5c2_s_sort,2)-size(j_rate_5c2_s_sort,2)*0.05);

%5c3
clear C
v = 1:1:5; C = nchoosek(v,3);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_5c3_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_5c3_s = cell2mat(j_rate_5c3_s);
j_rate_5c3_s_sort = sort(j_rate_5c3_s);
j_rate_5c3_95CI_l = j_rate_5c3_s_sort(:,size(j_rate_5c3_s_sort,2)*0.05);
j_rate_5c3_95CI_h = j_rate_5c3_s_sort(:,size(j_rate_5c3_s_sort,2)-size(j_rate_5c3_s_sort,2)*0.05);

%5c4
clear C
v = 1:1:5; C = nchoosek(v,4);
for j = 1:100
    for i = 1:size(C,1)
        A = RMSE_judge_s{:,j}(:,C(i,:));
        Asum = sum(A,2);
        j_rate_5c4_s{:,j}(:,i) = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
        clear A Asum
    end
end
clear i j
j_rate_5c4_s = cell2mat(j_rate_5c4_s);
j_rate_5c4_s_sort = sort(j_rate_5c4_s);
j_rate_5c4_95CI_l = j_rate_5c4_s_sort(:,size(j_rate_5c4_s_sort,2)*0.05);
j_rate_5c4_95CI_h = j_rate_5c4_s_sort(:,size(j_rate_5c4_s_sort,2)-size(j_rate_5c4_s_sort,2)*0.05);

%5c5
clear C
C = nchoosek(v,5);
for i = 1:100
    A = RMSE_judge_s{:,i}(:,C(1,:));
    Asum = sum(A,2);
    j_rate_5c5_s{:,i} = numel(find(Asum > 0)) / size(RMSE_judge_5c1,1)*100;
    clear A Asum
end
clear i 
j_rate_5c5_s = cell2mat(j_rate_5c5_s);
j_rate_5c5_s_sort = sort(j_rate_5c5_s);
j_rate_5c5_95CI_l = j_rate_5c5_s_sort(:,5);
j_rate_5c5_95CI_h = j_rate_5c5_s_sort(:,95);

j_rate_all = [j_rate j_rate_5c2 j_rate_5c3 j_rate_5c4 j_rate_5c5];
%J_rate_all = [{j_rate} {j_rate_5c2} {j_rate_5c3} {j_rate_5c4} {j_rate_5c5}]%保存用
Judge=[repmat({'5c1'},numel(j_rate),1);...
    repmat({'5c2'},numel(j_rate_5c2),1);...
    repmat({'5c3'},numel(j_rate_5c3),1);...
    repmat({'5c4'},numel(j_rate_5c4),1);repmat({'5c5'},numel(j_rate_5c5),1)];

CI95_h = [j_rate_s_95CI_h j_rate_5c2_95CI_h j_rate_5c3_95CI_h j_rate_5c4_95CI_h j_rate_5c5_95CI_h]';
CI95_l = [j_rate_s_95CI_l j_rate_5c2_95CI_l j_rate_5c3_95CI_l j_rate_5c4_95CI_l j_rate_5c5_95CI_l]';

CI95_h = [{j_rate_s_95CI_h} {j_rate_5c2_95CI_h} {j_rate_5c3_95CI_h} {j_rate_5c4_95CI_h} {j_rate_5c5_95CI_h}]';
CI95_l = [{j_rate_s_95CI_l} {j_rate_5c2_95CI_l} {j_rate_5c3_95CI_l} {j_rate_5c4_95CI_l} {j_rate_5c5_95CI_l}]';


% 保存用
CI95_H = [{j_rate_s_95CI_h} {j_rate_5c2_95CI_h} {j_rate_5c3_95CI_h} {j_rate_5c4_95CI_h} {j_rate_5c5_95CI_h}]';
CI95_L = [{j_rate_s_95CI_l} {j_rate_5c2_95CI_l} {j_rate_5c3_95CI_l} {j_rate_5c4_95CI_l} {j_rate_5c5_95CI_l}]';


figure;
ar = area([1 2 3 4 5],[CI95_l CI95_h]);hold on; 
boxplot(j_rate_all,Judge);hold on;
ylabel('prediction rate');xlabel('Number of MCs');
ar(1,2).FaceColor = '[0.7490 0.7490 0.7490]';
ar(1,2).EdgeColor = '[0.7490 0.7490 0.7490]';
ar(1,1).FaceColor = '[1 1 1]';
ar(1,1).EdgeColor = '[1 1 1]';

%プール後

cell1=[J_rate_all_5c1;J_rate_all_4c1;J_rate_all_3c1];
cell2=[J_rate_all_5c2;J_rate_all_4c2;J_rate_all_3c2];
cell3=[J_rate_all_5c3;J_rate_all_4c3;J_rate_all_3c3];
cell4=[J_rate_all_5c4;J_rate_all_4c4];
cell5=[J_rate_all_5c5];
All_cell=[cell1;cell2;cell3;cell4;cell5];
Judge=[repmat({'1cell'},numel(cell1),1);...
    repmat({'2cells'},numel(cell2),1);...
    repmat({'3cells'},numel(cell3),1);...
    repmat({'4cells'},numel(cell4),1);repmat({'5cells'},numel(cell5),1)];

judge=[repmat(1,numel(cell1),1);...
    repmat(2,numel(cell2),1);...
    repmat(3,numel(cell3),1);...
    repmat(4,numel(cell4),1);repmat(5,numel(cell5),1)];

% 
% figure;
% boxplot(All_cell,Judge);hold on;
% swarmchart(judge,All_cell);hold on;
% ar = area([1 2 3 4 5],[CI95_L_all_m CI95_H_all_m]);hold on; 
% figure;
% swarmchart(judge,All_cell,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
% boxchart(judge,All_cell,'LineWidth',1.5,'BoxEdgeColor','k','BoxFaceColor','w','BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');
% 
% ylabel('% predictable SW');xlabel('Number of MCs');
% ar(1,2).FaceColor = '[0.7490 0.7490 0.7490]';
% ar(1,2).EdgeColor = '[0.7490 0.7490 0.7490]';
% ar(1,1).FaceColor = '[1 1 1]';
% ar(1,1).EdgeColor = '[1 1 1]';

% figure 6c
% load '221129_prediction_rate_MCs.mat'
for i=1:6
y(i)=9.1803*i;
end
y=[0 y];
x=[0:6];
figure;
plot(x,y,'Color',[0 0.502 0.7529]); hold on;
swarmchart(judge,All_cell,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(judge,All_cell,'LineWidth',1.5,'BoxEdgeColor','k','BoxFaceColor','w','BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none'); hold on;ylim([0 50]);xlim([0.5 5.5]);
ylabel('% predictable SW');xlabel('Number of MCs');
% % one way ANOVA
% [p,tbl,stats]=anova1(All_cell,judge);

jt = [All_cell judge];
p = jttrend_ayako(jt); %modulate program for calculate precise p-value

% linear summation
% slope→　mean(cell1)=9.1803

%%

% % cosine similarity (digital)
% clear C v
% v=1:1:4; C=nchoosek(v,2);
% for i=1:size(C,1)
%     cs(i,:) = getCosineSimilarity(RMSE_judge_5c1(:,C(i,1)),RMSE_judge_5c1(:,C(i,2)));
% end
% 
% cs_all = [cs_all;cs];
% 
% clearvars -except cs_all
% 
% figure;
% histogram(cs_all,8);xlabel('Cosine Similarity');ylabel('Count');xlim([-1 1]);
% 
% cs_all_cosd = acosd(cs_all); % cosine similarity to angle(deg)
% cs_all_cosd_r = deg2rad(cs_all_cosd);% angle(deg) to radian
% 
% [pval,v] = circ_vtest(cs_all_cosd_r,circ_ang2rad(90)) % modulated calculating method for pval
% 
% figure;
% polarhistogram(cs_all_cosd_r,15,'FaceColor',[20/255,180/255,20/255]);thetalim([0 180]);

%% RMSEの相関係数
% 
% clear v C
% v=1:1:3; C=nchoosek(v,2);
% 
% for i=1:size(C,1)
%     rr{i}=corrcoef(RMSE_judge_3c1(:,C(i,1)),RMSE_judge_3c1(:,C(i,2)));
%     r(i,:)=rr{1,i}(1,2);
% end
% clear i
% r_all = [r_all;r];
% clearvars -except r_all
% 
% figure;
% histogram(r_all,14);xlabel('correlation coefficient');ylabel('Count');xlim([-.5 .5])
% 
% % 横方向にすべて値が0になっているrowを省いて相関係数を計算する
% RMSE_judge_4c1_original=RMSE_judge_4c1;
% for i=1:size(RMSE_judge_4c1,1)
%     if RMSE_judge_4c1(i,:) == zeros(1,4);
%         RMSE_judge_4c1(i,:)=[];
%     else
%     end
% end
% 
% % dot product
% for i=1:size(C,1)
%     D(i,:)=dot(RMSE_judge_4c1(:,C(i,1)),RMSE_judge_4c1(:,C(i,2)));
% end
% 
% % cosine similarity 
% % code URL;https://jp.mathworks.com/matlabcentral/fileexchange/62978-getcosinesimilarity-x-y
% for i=1:size(C,1)
%     cs(i,:) = getCosineSimilarity(RMSE_judge_4c1(:,C(i,1)),RMSE_judge_4c1(:,C(i,2)));
% end

%% vivo RMSE_judge
% Update 221031

RMSE_judge_vivo = [];
for i = 1 : size(RMSE_cv,1)
        if RMSE_cv(i) < RMSE_corr_cv_s_sort(i,5)
            RMSE_judge_vivo(i,:)=1;
        else
            RMSE_judge_vivo(i,:)=0;
        end
end
clear i j

prediction_rate = numel(find(RMSE_judge_vivo==1))/size(RMSE_judge_vivo,1)*100;

%%
% RMSE_corr_4c2_s1=RMSE_corr_4c2_s(:,1:100);
% RMSE_corr_4c2_s2=RMSE_corr_4c2_s(:,101:200);
% RMSE_corr_4c2_s3=RMSE_corr_4c2_s(:,201:300);
% RMSE_corr_4c2_s4=RMSE_corr_4c2_s(:,301:400);
% RMSE_corr_4c2_s5=RMSE_corr_4c2_s(:,401:500);
% RMSE_corr_4c2_s6=RMSE_corr_4c2_s(:,501:600);
% 
% RMSE_corr_4c2_s1_sort=sort(RMSE_corr_4c2_s1,2);
% RMSE_corr_4c2_s2_sort=sort(RMSE_corr_4c2_s2,2);
% RMSE_corr_4c2_s3_sort=sort(RMSE_corr_4c2_s3,2);
% RMSE_corr_4c2_s4_sort=sort(RMSE_corr_4c2_s4,2);
% RMSE_corr_4c2_s5_sort=sort(RMSE_corr_4c2_s5,2);
% RMSE_corr_4c2_s6_sort=sort(RMSE_corr_4c2_s6,2);
% 
% RMSE_corr_4c2_s_sort={RMSE_corr_4c2_s1_sort RMSE_corr_4c2_s2_sort RMSE_corr_4c2_s3_sort RMSE_corr_4c2_s4_sort RMSE_corr_4c2_s5_sort RMSE_corr_4c2_s6_sort};
% 
% RMSE_judge_4c2 = [];
% for i = 1 : size(RMSE_corr_4c2_s_sort,2)
%     for j = 1 : size(RMSE_corr_4c2,1)
%         if RMSE_corr_4c2(j,i) < RMSE_corr_4c2_s_sort{:,i}(j,5)
%             RMSE_judge_4c2(j,i)=1;
%         else
%             RMSE_judge_4c2(j,i)=0;
%         end
%     end
% end
% 

%%
% RMSE_corr_4c3_s1=RMSE_corr_4c3_s(:,1:100);
% RMSE_corr_4c3_s2=RMSE_corr_4c3_s(:,101:200);
% RMSE_corr_4c3_s3=RMSE_corr_4c3_s(:,201:300);
% RMSE_corr_4c3_s4=RMSE_corr_4c3_s(:,301:400);
% 
% RMSE_corr_4c3_s1_sort=sort(RMSE_corr_4c3_s1,2);
% RMSE_corr_4c3_s2_sort=sort(RMSE_corr_4c3_s2,2);
% RMSE_corr_4c3_s3_sort=sort(RMSE_corr_4c3_s3,2);
% RMSE_corr_4c3_s4_sort=sort(RMSE_corr_4c3_s4,2);
% 
% RMSE_corr_4c3_s_sort={RMSE_corr_4c3_s1_sort RMSE_corr_4c3_s2_sort RMSE_corr_4c3_s3_sort RMSE_corr_4c3_s4_sort};
% 
% RMSE_judge_4c3 = [];
% for i = 1 : size(RMSE_corr_4c3_s_sort,2)
%     for j = 1 : size(RMSE_corr_4c3,1)
%         if RMSE_corr_4c3(j,i) < RMSE_corr_4c3_s_sort{:,i}(j,5)
%             RMSE_judge_4c3(j,i)=1;
%         else
%             RMSE_judge_4c3(j,i)=0;
%         end
%     end
% end
%%
% RMSE_corr_s_sort=sort(RMSE_corr_s,2);
% RMSE_judge = [];
%     for j = 1 : size(RMSE_corr_s,1)
%         if RMSE_corr(j,:) < RMSE_corr_s_sort(j,5)
%             RMSE_judge(j,:)=1;
%         else
%             RMSE_judge(j,:)=0;
%         end
%     end

%%
% RMSE_corr_3c2_s1=RMSE_corr_3c2_s(:,1:100);
% RMSE_corr_3c2_s2=RMSE_corr_3c2_s(:,101:200);
% RMSE_corr_3c2_s3=RMSE_corr_3c2_s(:,201:300);
% 
% RMSE_corr_3c2_s1_sort=sort(RMSE_corr_3c2_s1,2);
% RMSE_corr_3c2_s2_sort=sort(RMSE_corr_3c2_s2,2);
% RMSE_corr_3c2_s3_sort=sort(RMSE_corr_3c2_s3,2);
% 
% RMSE_corr_3c2_s_sort={RMSE_corr_3c2_s1_sort RMSE_corr_3c2_s2_sort RMSE_corr_3c2_s3_sort};
% 
% RMSE_judge_3c2 = [];
% for i = 1 : size(RMSE_corr_3c2_s_sort,2)
%     for j = 1 : size(RMSE_corr_3c2,1)
%         if RMSE_corr_3c2(j,i) < RMSE_corr_3c2_s_sort{:,i}(j,5)
%             RMSE_judge_3c2(j,i)=1;
%         else
%             RMSE_judge_3c2(j,i)=0;
%         end
%     end
% end
