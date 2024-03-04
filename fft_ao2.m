%% fft_ao2.m
% 高速フーリエ変換のプログラム
% 50Hzまで表示　20151201
% スペクトラムのパワーは、
% http://www.sfn.org/~/media/SfN/Documents/Short%20Courses/2013%20Short%20Course%20II/SC2%20Kramer.ashx
% を参考にした。
%　Update 220930

function [Pmaxrgpower,PmaxrgHz,AUCsum]=fft_ao2(x,fs,t);
if size(x,2)>1 % 行列xが行ベクトルならば、xを転置させて、列ベクトルにする
    x=x';
end

if size(t,2)>1
    t=t';
end

X=x-mean(x);
L=length(t);
T=t(end);
Y=fft(X);
FQ=(0:L-1)'*(fs/L);
P=(Y.*conj(Y))*(2/(T*fs^2)); %FQと要素数同じ。値の求め方は、次のURL参照。http://www.sfn.org/~/media/SfN/Documents/Short%20Courses/2013%20Short%20Course%20II/SC2%20Kramer.ashx
minFQind=min(find(FQ>250)); %default=50（変更したらfigのaxisも変更する）→とりあえず50Hzまで表示する（FQ>50とする）けど、グラフを描くときにx軸の上限を限定すればいいだけだと思ったら計算遅くなるからやはり必要
% maxFQind=min(find(P==max(P))); %Powerが最大になるインデックス、かつ50Hz以下の点。FQとPの要素数が同じなので、Pを使ってfindしてもOK
FQfig=FQ(1:minFQind);
Pfig=P(1:minFQind);
% Pmax=Pfig(maxFQind);
rg = 13:1:27; % FQfigで120-250Hzのindex
[Pmaxrgpower,I] = max(Pfig(rg));
PmaxrgHz = FQfig(rg(I));

% AUCの計算
for i = 1 : numel(Pfig)-1
    AUC(i) = (Pfig(i) + Pfig(i+1))/(2*(FQfig(i+1)-FQfig(i)));
end
AUC = AUC';
rgforAUC = 13:1:26; 
AUCsum = sum(AUC(rgforAUC));
% 
% figure;
% plot(FQfig,Pfig,'k'); xlim([120 250]); hold on; %ylim([0 10e-6]);
% %plot(FQfig(maxFQind),Pmax,'ro'); xlim([80 200]);ylim([0 10e-5]); hold off;
% plot(FQfig(rg(I)),Pmaxrgpower,'ro'); xlim([120 250]); hold off; %ylim([0 10e-6]);
% xlabel('frequency (Hz)'); ylabel('Power ((unit of x)^2/Hz)');
end

