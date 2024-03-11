%% Vm_fluctuation_for_vivo3.m %%
%% 200310
%% Usage: difamp;ΔVm
%% Vm_fluctuation_for_vivo2('G:\vivo patch解析\MC\180219\pa350')



function [difamp,conf1,conf2]=Vm_fluctuation_for_vivo3(basepath);
cd (basepath);
[fname fpath]=uigetfile('*.atf','Select ATF file');
if isequal([fname fpath],[0 0])
    display('User canceled.');
    return;
end
cd(fpath);

%load matfile
matfile = dir(fullfile(basepath,'*.mat'));
if length(matfile) < 1; error('No mat file detected in the folder!'); return;
elseif length(matfile) == 1; matname = matfile.name;
else
    [matname,fpath]=uigetfile('*.mat','Select MAT file of ripple information');
    if isequal([matname fpath],[0 0])
        display('User canceled.');
        return;
    end
end
disp(strcat('Loading:',matname));
load(matname);

% Import ATF file
[a,b,c,d]=import_atf(fname); %
t=d(:,1);
lfp=d(:,3); vm=d(:,2);
fs=round(1/(t(2)-t(1)));
dif=t(1); 
t=t-dif; 
clear a b c d

% import information of ripples（from AllResults obtained from 'ripple_finalization2.m'）
%rippleinfo=AllResults.ripplefinaltiming; %AllResultsの構造体からリップルの情報をとりだす
ripplepeakind=AllResults.RippleInfo2(:,3);
ripplenum=size(ripplepeakind,1);
ripplepeakind=ripplepeakind-dif;
rippleindex=AllResults.ripplefinalindex2;


L=0.1;
 for k=1:ripplenum
  rippleind(k,:)=[ripplepeakind(k)-round(0.3*L*fs):ripplepeakind(k)+round(0.4*L*fs)];
 end
 clear k
 rippleind=round(rippleind);
 vmall=vm(rippleind);
 %tall=t(rippleind);
for i=1:ripplenum
    % detect depolarization
    maxV(i,:)=max(vmall(i,:)); 
    F=find(vmall(i,:)==maxV(i,:));

   % if maximum ΔVm detected more than two,select nearest ΔVm point near to SW peak
    if size(F,2) >= 2
        suitF(i,:)=F(knnsearch(rippleind(1,F)',ripplepeakind(i))); 
    elseif size(F,2) == 1
        suitF(i,:)=F;
    end 

    Vmaxidx(i,:)=rippleind(i,suitF(i));

    % if max ΔVm detected at the edge of the window,onset Vm is defined before 50ms
    % In general, the lowest Vm between the max Vm (~-50ms) is the min Vm
%for i=1:ripplenum
    if  Vmaxidx(i,1)==rippleind(i,1); 
        epsppeak(i,:)=[Vmaxidx(i)-round(0.5*L*fs):Vmaxidx(i)];
        epsp(i,:)=vm(epsppeak(i,:));
        mn(i,1)=epsp(i,1); %min Vm
    elseif Vmaxidx(i,1)==rippleind(i,end);
        epsppeak(i,:)=[Vmaxidx(i)-round(0.5*L*fs):Vmaxidx(i)];
        epsp(i,:)=vm(epsppeak(i,:));
        mn(i,1)=min(epsp(i,:));%changed from % mn(i,1)=epsp(i,end)(200310)
    else
        epsppeak(i,:)=[Vmaxidx(i)-round(0.5*L*fs):Vmaxidx(i)];
        epsp(i,:)=vm(epsppeak(i,:));
        mn(i,1)=min(epsp(i,:)); %min Vm
    end
%end
%for i=1:ripplenum
    % depolarization
    n(i,1)=min(find(epsp(i,:)==mn(i,1))); %idx of min Vm
    nn=epsppeak(i,:);
    nnn=n(i,:);
    ep(i,1)=(nn(nnn)); % index(for confirmation)   
    epspamp(i,1)=maxV(i)-mn(i); %max-min=Vm %detect ΔVm using the min value as baseline within -20ms from peak
    
    % hyperpolarization
    minV(i,:)=min(vmall(i,:));
    minF=find(vmall(i,:)==minV(i,:));
    % if min ΔVm detected more than two,select nearest ΔVm point near to SW peak
    if size(minF,2) >= 2
        minsuitF(i,:)=minF(knnsearch(rippleind(1,minF)',ripplepeakind(i,:)));           
    elseif size(minF,2) == 1
        minsuitF(i,:)=minF;
    end 
    Vminidx(i,:)=rippleind(i,minsuitF(i));

    % if min Vm is detected at the left edge of the window, ΔVm is set to 0
    if  Vminidx(i,1)~=rippleind(i,1); 
        ipspbase(i,:)=[Vminidx(i)-round(0.5*L*fs):Vminidx(i)];
        ipsp(i,:)=vm(ipspbase(i,:));
        mmn(i,1)=max(ipsp(i,:)); %minVm
    else
        ipspbase(i,:)=[Vminidx(i)-round(0.5*L*fs):Vminidx(i)];
        ipsp(i,:)=vm(ipspbase(i,:));
        mmn(i,1)=ipsp(i,1);
    end
    
    % hyperpolarization
    minn(i,1)=min(find(ipsp(i,:)==mmn(i,1)));
    minnn=ipspbase(i,:);
    minnnn=minn(i,:);
    ip(i,1)=(minnn(minnnn)); % index(for confirmation)
    ipspamp(i,1)=minV(i)-mmn(i);
    
    if abs(epspamp(i,1)) > abs(ipspamp(i,1))
        difamp(i,1)=epspamp(i,1);
        allp(i,1)=ep(i,1);
        allf(i,1)=Vmaxidx(i,1);
    else
        difamp(i,1)=ipspamp(i,1);
        allp(i,1)=ip(i,1);
        allf(i,1)=Vminidx(i,1);
    end
end
%ratio of dep and hyp
% dep=sum(difamp>0 & difamp<20); hyp=sum(difamp<0 & difamp>-20);
% firing=sum(abs(difamp)>=20);
% depratio=dep/ripplenum*100; hypratio=hyp/ripplenum*100; firingratio=firing/ripplenum*100;

%check figure
SS = get(0, 'ScreenSize'); 
    figure('Position',SS);
    subplot(211),plot(t,vm,'k',...
        t(allf),vm(allf),'ro',...
        t(allp),vm(allp),'gs','MarkerSize',5);
    subplot(212),plot(t,lfp,'k'); hold on;
    plot(t,lfp,'k',t(ripplepeakind),lfp(ripplepeakind),'ro','MarkerSize',5);
    for i=1:ripplenum
     plot(t(rippleindex(i,2):rippleindex(i,4)),lfp(rippleindex(i,2):rippleindex(i,4)),'m');
    end    
   
%end
    conf1=t(allf); conf2=t(allp);
    
end
