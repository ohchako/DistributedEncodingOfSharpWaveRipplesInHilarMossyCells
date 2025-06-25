%% Figures_revised.m
%% Update; 250625 for revised ver
%% details of variable is written in each program
%% Figure.1,2

% Fig.1C
% load Fig.1C_data (original data from data4cells(invitro)→190823_3)
% peak50ms → raw SW data (±50ms from SW peak)
% peak50ms_n → rescaled from [0 1] , used for machine learning
% representative trace; #430,#605,#678,#851

for i=1:size(peak50ms,1)
    for j=1:size(peak50ms,1)
        RMSE_xtest(i,j) = sqrt(immse(peak50ms(i,:),peak50ms(j,:)));
    end
end
clear i j

Y = mdscale(RMSE_xtest,2,'criterion','metricstress');
figure;
plot(Y(:,1),Y(:,2),'k.'); hold on; plot(Y(851,1),Y(851,2),'r.','MarkerSize',10);



% Fig.1D
% load Fig.1D_data
% difampr; calculated from Vm_fluctuation5.m

difamp = difampr';
figure;
imagesc(difamp);colormap(jet);
set(gca,'XDir','normal');
xlabel('SWR #');ylabel('Cell #');
xticks([50 100]);yticks([1 2 3 4 5]);
colormap(cmap);caxis([-5 15]);colorbar

% Fig.1E,2D
%load Fig.1E_data or Fig.2D_data

figure;
scatter(allSW_z,allVm_z,'.');
xlabel('SW amplitude (Z-scored)');ylabel('ΔVm (Z-scored)');

%% Figure.3

% Fig.3C
% load Fig3C_data

% [f,x]=ecdf(RMSE_real);
% [F,X]=ecdf(RMSE_shuffle);
% [h,p,k]=kstest2(x,X)
% 
% figure;
% plot(x,f);hold on;plot(X,F);
% ylabel('cumulative probability');xlabel('RMSE');title('5 cells cumulative');xlim([0 0.25]);
% legend('real','shuffle','Location','southeast');

%250307
%new Fig.3C
%load Fig.3C_data_revised
M_real = RMSE_corr_cv_5c5;
M_shuffle = mean(RMSE_corr_s_5c5,2);

M_all = [M_real;M_shuffle];
str1 = "real"; str2 = "shuffle";
P = repelem([str1,str2],[178,178]); % 178 SWR events
PP = categorical(P)';

figure;
% swarmchart(PP(1:178,1),M_all(1:178,1),'filled','MarkerFaceColor',[0.07 0.71 0.07],'SizeData',20);hold on; ylabel('RMSE')
% swarmchart(PP(179:end,1),M_all(179:end,1),'filled','MarkerFaceColor',[0.6 0.6 0.6],'SizeData',20);hold on; ylim([0.04 0.26])
boxchart(PP(1:178,1),M_all(1:178,1),'LineWidth',1.5,'BoxEdgeColor','k','BoxFaceColor',[20/255 180/255 20/255],'WhiskerLineColor','k','BoxFaceAlpha',0.5,'BoxWidth',0.3,'MarkerStyle','none');ylim([0.04 0.24]); hold on
boxchart(PP(179:end,1),M_all(179:end,1),'LineWidth',1.5,'BoxEdgeColor','k','BoxFaceColor',[128/255 128/255 128/255],'WhiskerLineColor','k','BoxFaceAlpha',0.5,'BoxWidth',0.3,'MarkerStyle','none');%'BoxFaceColor','w',



% Fig.3D
% load Fig3D_data

% RMSE_cv_1cell_m=mean(RMSE_cv_1cell);
% RMSE_cv_2cell_m=mean(RMSE_cv_2cell);
% RMSE_cv_3cell_m=mean(RMSE_cv_3cell);
% RMSE_cv_4cell_m=mean(RMSE_cv_4cell);
% RMSE_cv_5cell_m=mean(RMSE_cv_5cell);

M = [RMSE_cv_1cell_m RMSE_cv_2cell_m RMSE_cv_3cell_m]';
M_z = zscore(M);

PP = [repmat(1,numel(RMSE_cv_1cell_m),1);...
    repmat(2,numel(RMSE_cv_2cell_m),1);...
    repmat(3,numel(RMSE_cv_3cell_m),1)];
% concatenate all data
M_z_all=[M_z_all;M_z];
PP_all=[PP_all;PP];

figure;
swarmchart(PP_all,M_z_all,'filled','MarkerFaceColor',[0.6 0.6 0.6],'SizeData',20);hold on;
boxchart(PP_all,M_z_all,'LineWidth',1.5,'BoxEdgeColor','k','BoxFaceColor','w','BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');

% Fig.3E
% load Fig3E_data_revised
sz=250;
figure;
yyaxis right
semilogy(RMSE(:,1),RMSE(:,3),'color',[255/255 70/255 50/255]);hold on;
scatter(RMSE(:,1),RMSE(:,3),sz,'.','MarkerEdgeColor',[255/255 70/255 50/255],'MarkerFaceColor',[255/255 70/255 50/255]);hold on;
scatter(RMSE_vivo(1),RMSE_vivo(3),sz,'.','MarkerEdgeColor',[255/255 70/255 50/255],'MarkerFaceColor',[255/255 70/255 50/255]);hold on;
plot(RMSE(:,5),RMSE(:,4),'--','color',[255/255,70/255,50/255]);set(gca,'YDir','reverse');xlim([0.5 7.5]);ylim([10^(-100) 10^6]);hold on;
yyaxis left
plot(RMSE(:,1),abs(RMSE(:,2)),'k');hold on;scatter(RMSE(:,1),abs(RMSE(:,2)),sz,'k','.');ylim([0 25]);hold on
scatter(RMSE_vivo(1),abs(RMSE_vivo(2)),sz,'k','.');hold off



% MC_real1=[MC_real1;RMSE_cv_1cell_re];MC_real2=[MC_real2;RMSE_cv_2cell_re];MC_real3=[MC_real3;RMSE_cv_3cell_re];%MC_real4=[MC_real4;RMSE_cv_4cell_re];
% clear A A_reshape
% %A=[mean(RMSE_corr_cv_s_5c2(:,1:10),2) mean(RMSE_corr_cv_s_5c2(:,11:20),2) mean(RMSE_corr_cv_s_5c2(:,21:30),2) mean(RMSE_corr_cv_s_5c2(:,31:40),2) mean(RMSE_corr_cv_s_5c2(:,41:50),2) mean(RMSE_corr_cv_s_5c2(:,51:60),2) mean(RMSE_corr_cv_s_5c2(:,61:70),2) mean(RMSE_corr_cv_s_5c2(:,71:80),2) mean(RMSE_corr_cv_s_5c2(:,81:90),2) mean(RMSE_corr_cv_s_5c2(:,91:100),2)];
% %A_reshape=reshape(A,[],1);
% % A=[mean(RMSE_corr_cv_s_5c4(:,1:10),2) mean(RMSE_corr_cv_s_5c4(:,11:20),2) mean(RMSE_corr_cv_s_5c4(:,21:30),2) mean(RMSE_corr_cv_s_5c4(:,31:40),2) mean(RMSE_corr_cv_s_5c4(:,41:50),2)];
% 
% A=[mean(RMSE_corr_cv_s_3c2(:,1:10),2) mean(RMSE_corr_cv_s_3c2(:,11:20),2) mean(RMSE_corr_cv_s_3c2(:,21:30),2)];
% A_reshape=reshape(A,[],1);
% clearvars -except MC_real1 MC_real2 MC_real3 MC_real4 MC_real5 MC_shuffle1 MC_shuffle2 MC_shuffle3 MC_shuffle4 MC_shuffle5 
% 
% load('RMSEs.mat')
% MC_real1=[MC_real1;RMSE_cv_1cell_re];MC_real2=[MC_real2;RMSE_cv_2cell_re];MC_real3=[MC_real3;RMSE_cv_3cell_re];%MC_real4=[MC_real4;RMSE_cv_4cell_re];
% %mean(RMSE_corr_cv_s_4c4,2);
% %MC_shuffle4=[MC_shuffle4;ans];
% A=[mean(RMSE_corr_cv_s_4c1(:,1:10),2) mean(RMSE_corr_cv_s_4c1(:,11:20),2) mean(RMSE_corr_cv_s_4c1(:,21:30),2) mean(RMSE_corr_cv_s_4c1(:,31:40),2)];
% A_reshape=reshape(A,[],1);
% MC_shuffle1=[MC_shuffle1;A_reshape];
% clear A A_reshape
% A=[mean(RMSE_corr_cv_s_4c2(:,1:10),2) mean(RMSE_corr_cv_s_4c2(:,11:20),2) mean(RMSE_corr_cv_s_4c2(:,21:30),2) mean(RMSE_corr_cv_s_4c2(:,31:40),2) mean(RMSE_corr_cv_s_4c2(:,41:50),2) mean(RMSE_corr_cv_s_4c2(:,51:60),2)];
% A_reshape=reshape(A,[],1);
% MC_shuffle2=[MC_shuffle2;A_reshape];
% clear A A_reshape
% A=[mean(RMSE_corr_cv_s_4c3(:,1:10),2) mean(RMSE_corr_cv_s_4c3(:,11:20),2) mean(RMSE_corr_cv_s_4c3(:,21:30),2) mean(RMSE_corr_cv_s_4c3(:,31:40),2)];
% A_reshape=reshape(A,[],1);
% MC_shuffle3=[MC_shuffle3;A_reshape];
% clearvars -except MC_real1 MC_real2 MC_real3 MC_shuffle1 MC_shuffle2 MC_shuffle3
% 
% MC_real1=[MC_real1;RMSE_cv_1cell_re];MC_real2=[MC_real2;RMSE_cv_2cell_re];MC_real3=[MC_real3;RMSE_cv_3cell_re];%MC_real4=[MC_real4;RMSE_cv_4cell_re];
% %mean(RMSE_corr_cv_s_4c4,2);
% %MC_shuffle4=[MC_shuffle4;ans];
% A=[mean(RMSE_corr_cv_s_3c1(:,1:10),2) mean(RMSE_corr_cv_s_3c1(:,11:20),2) mean(RMSE_corr_cv_s_3c1(:,21:30),2)];
% A_reshape=reshape(A,[],1);
% MC_shuffle1=[MC_shuffle1;A_reshape];
% clear A A_reshape
% A=[mean(RMSE_corr_cv_s_3c2(:,1:10),2) mean(RMSE_corr_cv_s_3c2(:,11:20),2) mean(RMSE_corr_cv_s_3c2(:,21:30),2)];
% A_reshape=reshape(A,[],1);
% MC_shuffle2=[MC_shuffle2;A_reshape];
% clear A A_reshape
% mean(RMSE_corr_cv_s_3c3,2);
% MC_shuffle3=[MC_shuffle3;ans];
% clearvars -except MC_real1 MC_real2 MC_real3 MC_shuffle1 MC_shuffle2 MC_shuffle3
%% Figure.4
% load Fig.4_data
% Fig.4A
figure;
histogram(prediction_rate_1cell,11)

% fig4B polarplot
figure;
for i=1:size(R,1)
pl = polarplot(R(i,:),rho(i,:),'.');thetalim([0 180]);rlim([0 1]);hold on;
pl.Color = RGB_stack(i,:);
pl.MarkerSize = 12;
end

% fig.4C
figure;
histogram(H_sum_s)

% Fig.4D,E
figure;plot(prediction_rate_1cell,Cm,'.')
figure;plot(prediction_rate_1cell,Rm,'.')
figure;plot(prediction_rate_1cell,EPSPamp,'.')
figure;plot(prediction_rate_1cell,EPSPfrq,'.')

%% Figure.5

% Fig.5A left
figure;
subplot(2,2,1);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,1),color);title('cell #1');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,2);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,2),color);title('cell #2');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,3);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,3),color);title('cell #3');xlabel('MDS1');ylabel('MDS2');
subplot(2,2,4);gscatter(Y(:,1),Y(:,2),RMSE_judge(:,4),color);title('cell #4');xlabel('MDS1');ylabel('MDS2');
% Fig.5A right is obtained from variable from MDSentropy2.m

% Fig.5B
figure;
semilogy(cellnum,pvalue_all_sort,'.');set(gca,'YDir','reverse')

% Fig.5C
[x,f]=ecdf(r_all);
figure;plot(f,x); xlabel('Correlation coefficient');ylabel('Cumulative probability');

%% Figure.6
% Fig.6A
% load Fig.6AB_data
figure;
subplot(2,1,1)
imagesc(RMSE_judge_5c1');colormap(CMAP);set(gca,'XDir','normal');set(gca,'YDir','normal');colorbar;
subplot(2,1,2)
imagesc(color');colormap(CMAP);colorbar;

% Fig.6B
% programs of venn diagram is downloaded from ↓
% https://jp.mathworks.com/matlabcentral/fileexchange/98974-venn-euler-diagram?tab=discussions
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

% Fig.6C
% load Fig6C_data
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

% Fig.6D
figure;
histogram(overlap_all);xlabel('Overlap % per total');ylabel('Count')

% Fig.6E 
figure;
plot(X-1,yCalc1,'Color',[0.74 0.74 0.74]);hold on;
swarmchart(P,SW_amp,'filled','MarkerFaceColor',[0.74 0.74 0.74],'SizeData',20);hold on;
boxchart(P,SW_amp,'BoxEdgeColor','k','BoxFaceColor','w','LineWidth',1.5,'BoxFaceAlpha',1,'BoxWidth',0.3,'MarkerStyle','none');ylim([-2 2.5]);title('SW amplitude (Z-scored)')


%% supple
% examine mixed-effect model
%250309
load('RMSEs.mat')
% cell1_R=RMSE_cv_1cell(:,1);cell2_R=RMSE_cv_1cell(:,2);cell3_R=RMSE_cv_1cell(:,3);
% cell1_s=RMSE_corr_cv_s_3c1_100s(:,1:100);cell2_s=RMSE_corr_cv_s_3c1_100s(:,101:200);cell3_s=RMSE_corr_cv_s_3c1_100s(:,201:300);

for i=1:size(RMSE_corr_cv_5c1,2)
    M(i,:) = mean(mean(RMSE_corr_cv_s_5c1_100s(:,1+(i-1)*100:i*100),2))-mean(RMSE_corr_cv_5c1(:,i));
end

save('Δshuf-real','M');
figure;
plot(ID_all,M_all,'k.','MarkerSize',20);hold on;
plot(ID_all,ID2,'.','MarkerSize',20);hold on;
plot(ID_all,ID3,'.','MarkerSize',20);hold on;
plot(ID_all,ID4,'.','MarkerSize',20);hold on;
xlim([0 15]);yline(0);xlabel('Mouse ID');ylabel('ΔRMSE(shuffle-real)');ylim([-0.005 0.013])