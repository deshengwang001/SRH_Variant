function [f0,VAD] = SRH_Variant(wave,Fs,edge)

% INPUTS:
%     - wave: audio signal
%     - Fs:   sampling frequency (Hz)
%     - edge: minimum and the maximum of pitch search range [minimum maximum]
% OUPUTS:
%     - f0:   vector of F0 values (with an hopsize of 10ms)
%     - VAD:  vector of voiced/unvoiced decisions (with an hopsize of 10ms)
% Note that the output sequence should be manually aligned to the real sequence. 
% For the Keele dataset, and the frame length in this code, the suggested offset is 6 frames 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is a variant of the following approach.
% T.Drugman, A.Alwan, "Joint Robust Voicing Detection and Pitch Estimation
% Based on Residual Harmonics", Interspeech11, Firenze, Italy, 2011

% We modify the SRH formula and add the pitch segment expansion operation.
% We acknowledge the authors for the origional SRH codes, 
% some of which serves as a starting point for this code.

% Matlab version used for developing this code: 2019a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% step 0 preparing 
if Fs > 22050
    wave=resample(wave,16000,Fs);
    Fs=16000;
end

Delta_threshold = 0.11;
Length_threshold = 5;

%% step 1
LPCorder=round(3/4*Fs/1000);
residual_Spectral = cal_LPC(wave,round(25/1000*Fs),round(5/1000*Fs),LPCorder);

%% step 2
[f0_candicate,SRH_Val] = SRH_Core(residual_Spectral',Fs,edge,1);

%% step 3
f0_median = median(f0_candicate(SRH_Val>0.1));

%% step 4
edge(1) = max(round(0.5*f0_median),edge(1));    %update low end f0
edge(2) = min(round(2*f0_median),edge(2));      %update high end f0

%% step 5
[f0_candicate,SRH_Val] = SRH_Core(residual_Spectral',Fs,edge,2);

%% step 6
f0_Row1  = f0_candicate(1,1:end);               %get the initial sequence
f0_num   = length(f0_Row1);

idx_Stop = 1;                                   %forward index
ii = 1;
while ii < f0_num - 1                           %search all the initial sequence
    idx_Start = ii;
    idx_End   = ii + 1;
    while idx_End <= f0_num && ( abs( f0_Row1(idx_End) - f0_Row1(idx_End-1) )/( f0_Row1(idx_End-1) + eps) < Delta_threshold ) %find the segment
        idx_End = idx_End + 1;
    end
    idx_End = idx_End - 1;                      %get the segment end

    if idx_End - idx_Start > Length_threshold   %check the length to determine this is a main segment
        % forward extension
        idx_Start = idx_Start - 1; 
        while idx_Start > idx_Stop && sum( abs( f0_candicate(:,idx_Start) - f0_Row1(idx_Start+1) )./f0_Row1(idx_Start+1) < Delta_threshold ) > 0          %stop condition: noone in the candidate is within delta_threshold
            idx = find ( abs( f0_candicate(:,idx_Start) - f0_Row1(idx_Start+1) ) == min( abs( f0_candicate(:,idx_Start) - f0_Row1(idx_Start+1) ) ) );     %find the one with the smallest shift
            f0_Row1(idx_Start) = f0_candicate(idx(1),idx_Start);                                                                                          %update f0_Row1
            idx_Start = idx_Start - 1; 
        end
        idx_Start = idx_Start + 1; 
        
        % backward extension
        idx_End = idx_End + 1; 
        while idx_End <= f0_num - 1 && sum( abs( f0_candicate(:,idx_End) - f0_Row1(idx_End-1) )./f0_Row1(idx_End-1) < Delta_threshold ) > 0               %stop condition: noone in the candidate is within delta_threshold
            idx = find ( abs( f0_candicate(:,idx_End) - f0_Row1(idx_End-1) ) == min( abs( f0_candicate(:,idx_End) - f0_Row1(idx_End-1) ) ) );             %find the one with the smallest shift
            f0_Row1(idx_End) = f0_candicate(idx(1),idx_End);                                                                                              %update f0_Row1
            idx_End = idx_End + 1; 
        end
        idx_End = idx_End - 1; 
        
        idx_Stop = idx_End;  %update the stop index for searching pitch segment
    end

    ii = idx_End + 1;        %update the start index for searching pitch segment
end
f0 = f0_Row1;                %this is the final updated sequence

%% step 7
f0 = movmedian(f0,3);

%% step 8
%%%%%%%%%%%%%%%%%%%%%%%%%%%% orignal VAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VAD = zeros(1,f0_num);                          %init VAD as (0:unvoiced), (1:voiced) will be updated later
VAD_Threshold = 0.07;
if std(SRH_Val)>0.05
    VAD_Threshold=0.085;
end
SEL= SRH_Val>VAD_Threshold;
VAD(SEL)=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% new added for VAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ii = 1;
while ii < f0_num - 1                           %search all the sequence
    idx_Start = ii;
    idx_End   = ii + 1;
    while idx_End <= f0_num && VAD(idx_End)     %find the segment
       idx_End = idx_End + 1;
    end
    idx_End = idx_End - 1;                      %get the segment end
    
    if idx_End - idx_Start <= Length_threshold   %check the length to determine this is a main segment
        VAD(idx_Start:idx_End) = 0;
    end
    
    ii = idx_End + 1;        %update the start index for searching pitch segment
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f0_candicate,SRH_Val] = SRH_Core(sig,Fs,edge,Num)
stop=2048;
hopSize=200;
deltaF = 2;

frameNum=floor((length(sig)-stop)/hopSize) + 1;
f0_candicate = zeros(Num,frameNum);
SRH_Val = zeros(1,frameNum);

start=1;
index=1;
BlackWin=blackman(stop-start+1);

while stop<=length(sig)
    sigFrame=sig(start:stop);
    sigFrame=sigFrame.*BlackWin;    
    sigFrame=sigFrame-mean(sigFrame);

    Spec=fft(sigFrame,Fs);
    Spec=abs(Spec(1:Fs/2));    
    Spec=Spec/sqrt(sum(Spec.^2));
        
    SRHs=zeros(1,edge(2));    
    
    % revise harmonic structure modeling method: strict integer multiple to loose constraint 
    for freq=edge(1):edge(2)
        SRHs(freq)= (max( Spec( 1*freq-deltaF:1*freq+deltaF ) ) + ...
                     max( Spec( 2*freq-deltaF:2*freq+deltaF ) ) + ...
                     max( Spec( 3*freq-deltaF:3*freq+deltaF ) ) + ...
                     max( Spec( 4*freq-deltaF:4*freq+deltaF ) ) + ...
                     max( Spec( 5*freq-deltaF:5*freq+deltaF ) ))- ...
                     (Spec(round(1.5*freq))+Spec(round(2.5*freq))+Spec(round(3.5*freq))+Spec(round(4.5*freq)));
    end
   
    SRHs(1:edge(1)-1) = min(SRHs); 
    SRHs = SRHs - min(SRHs)+eps; 
    
    [posi] = get_Peak(SRHs(1,1:min(size(SRHs,2),400)),1,0,[edge(1),edge(2)],Num);
    F0frame=posi(:,1);
    if Num == 2
        if size(F0frame,1) == 1
            F0frame(2) = F0frame(1);
        end
    end
    f0_candicate(:,index)=F0frame;
    SRH_Val(index)=SRHs(F0frame(1));
    
    start=start+hopSize;
    stop=stop+hopSize;
    index=index+1;
end

function [peakFreq] = get_Peak(x,MPD,MHD,edge,nPeak)
Pxx = x;
num_Pxx = length(Pxx);
MPH   = 0.0001*max(Pxx);
peakFreq  = NaN(nPeak,3);               %data inside:[peak center,left end,right end]

if Pxx(edge(1)) >= Pxx(edge(1)+1)       %cut uphill segment at the beginning
    idx_Right = edge(1);      
    while Pxx(idx_Right) >= Pxx(idx_Right+1)
        idx_Right = idx_Right + 1;
    end
    idx_Right = idx_Right - 1;
    Pxx(1:idx_Right) = nan;
end

if Pxx(end) >= Pxx(end-1)               %cut downhill segment at the end
    idx_Left = num_Pxx;  
    while Pxx(idx_Left) >= Pxx(idx_Left-1)
        idx_Left = idx_Left - 1;
    end
    idx_Left = idx_Left + 1;
    Pxx(idx_Left:end) = nan;
end

b = 1; % peak information update index
while b <= nPeak && sum(abs(Pxx(~isnan(Pxx)))) > 0              %search peak one by one, find one and romove one
    [temp_Peak,temp_Loc] = max( Pxx ); 
    if temp_Loc == 1 || temp_Loc == num_Pxx                     %1 check end 
        Pxx(temp_Loc) = nan;
    elseif isnan(Pxx(temp_Loc-1))                               %2 check leftside empty, right roll down
        idx_Right = temp_Loc + 1;
        while idx_Right <= num_Pxx && Pxx(idx_Right-1) >= Pxx(idx_Right) % roll down slope to right valley
          idx_Right = idx_Right + 1;
        end
        idx_Right = idx_Right - 1;
        
        Pxx(temp_Loc:idx_Right) = nan;
    elseif isnan(Pxx(temp_Loc+1))                               %2 check rightside empty, left roll down
        idx_Left = temp_Loc - 1;
        while idx_Left > 0 && Pxx(idx_Left) <= Pxx(idx_Left+1)               % roll down slope to left
          idx_Left = idx_Left - 1;
        end
        idx_Left = idx_Left + 1;
        
        Pxx(idx_Left:temp_Loc) = nan;
    else
        if temp_Peak > MPH && ...                               %3 first check if the power is greater than the threshold
            Pxx(temp_Loc) >= Pxx(temp_Loc-1) && Pxx(temp_Loc) >= Pxx(temp_Loc+1)
            
            idx_Left = temp_Loc - 1;                                         % sidelobes treated as noise
            while idx_Left > 0 && Pxx(idx_Left) <= Pxx(idx_Left+1)           % roll down slope to left
              idx_Left = idx_Left - 1;
            end
            idx_Left = idx_Left+1;                                           % provide indices to the peak border (inclusive)
            
            idx_Right = temp_Loc + 1;
            while idx_Right <= num_Pxx && Pxx(idx_Right-1) >= Pxx(idx_Right) % roll down slope to right
              idx_Right = idx_Right + 1;
            end
            idx_Right = idx_Right-1;                                         % provide indices to the peak border (inclusive)

            if Pxx(temp_Loc) > Pxx(idx_Left) && Pxx(temp_Loc) > Pxx(idx_Right) 
                if b > 1
                    if min( abs( temp_Loc - peakFreq(1:b-1,1) ) ) > MPD || min( abs( Pxx(temp_Loc) - peakPow(1:b-1,2) ) ) > MHD
                        peakFreq(b,:) = [temp_Loc,idx_Left,idx_Right];
                        b = b + 1;
                    end
                else
                    peakFreq(b,:) = [temp_Loc,idx_Left,idx_Right];
                    b = b + 1;
                end
            end
            idxToRemove = idx_Left:idx_Right;
            Pxx(idxToRemove) = nan;
        else
            break; 
        end
    end       
end
b = b - 1;

if b > 0
    peakFreq = peakFreq(1:min(nPeak,b),:);
else                  
    peakFreq = 0;
end

function [residual_Spectral] = cal_LPC(wave,frameLength,hopSize,LPCorder)

HannWin = hanning(frameLength+1);
residual_Spectral=zeros(1,length(wave));

start=1;
stop=start+frameLength;   

while stop<length(wave)
    sigFrame=wave(start:stop);
    sigFrame=sigFrame.*HannWin;

    [A,~]=lpc(sigFrame,LPCorder);

    inv=filter(A,1,sigFrame);
    inv=inv*sqrt(sum(sigFrame.^2)/sum(inv.^2));
    residual_Spectral(start:stop)=residual_Spectral(start:stop)+inv';

    start=start+hopSize;
    stop=stop+hopSize;
end

residual_Spectral=residual_Spectral/max(abs(residual_Spectral));

