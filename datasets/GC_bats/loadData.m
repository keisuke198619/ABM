% loadData
clear; close all
filenames = {'1-3001-3300frame','4-3065-3365frame'} ;
mat_dir = '.\'; % data_mat\
video_dir = '.\videos\';

% parameters
dim_xy = 3 ;
Order = 2 ; % order of filter
Fc = 5 ; % cut-off frequency
dbstop if error

if 1 % load data
    for f = 1:length(filenames)
        [rawdata,rawstr] = xlsread(['.\',filenames{f},'\',filenames{f},'.xlsx']);
        [label_raw,following] = xlsread('äeå¬ëÃÇÃí«îˆèÛë‘.xlsx',f);
        following = following(2:end,2:4) ;
        followed = and(strcmp(following(:,1),'ÅZ'),strcmp(following(:,2),'ÅZ')) ;
        K_max = length(followed) ;
        
        Fs = 1/((rawdata(end,1)-rawdata(end-1,1))/1000) ;
        rawdata = rawdata(7:end,2:end) ;
        [b,a] = butter(Order, Fc/(Fs/2), 'low') ; % filter
        
        T = size(rawdata,1) ; % effecitve time length
        K = sum(followed); % effective individuals
        
        kks = [1 sum(label_raw==1)+1] ;
        clear loc loc_filt vel vel_filt loc_nan
        for k = 1:K_max
            if followed(k) == 1 && (label_raw(k)==1 || label_raw(k)==2)
                kk = kks(label_raw(k)) ;
                label(:,kk) = label_raw(k) ;
                rawNo(kk,1) = k ;
                
                loc(:,:,kk) = rawdata(1:T,(k-1)*dim_xy+1:k*dim_xy) ; % time,xyz,agents
                % filter and compute velocity
                loc_filt(:,:,kk) = nanfilt(loc(:,:,kk),b,a,Order) ;
                vel(:,:,kk) = (loc(2:end,:,kk) - loc(1:end-1,:,kk))*Fs ;
                vel_filt(:,:,kk) = (loc_filt(2:end,:,kk) - loc_filt(1:end-1,:,kk))*Fs ;
                
                % kk = kk + 1 ;
                kks(label_raw(k)) = kks(label_raw(k)) + 1 ;
            end
        end
        % align position and velocity
        loc = loc(1:end-1,:,:) ;
        loc_filt = loc_filt(1:end-1,:,:) ;
        
        loc_nan = loc_filt ;
        % nan detection
        data_nan = vel_filt(:,1,:) ;
        data_nan(~isnan(data_nan)) = 0 ;
        data_nan(isnan(data_nan)) = 1 ;
        
        loc_nan(isnan(data_nan)) = NaN ;
        
        % padding of starting and ending values
        kk = 1 ;
        for k = 1:K_max
            if ~isempty(find(k==rawNo,1)) % followed(k) == 1
                if data_nan(1,:,kk) == 1
                    en = find(data_nan(:,1,kk)==0,1) ;
                    vel(1:en,:,kk) = repmat(vel(en+1,:,kk),en,1,1) ;
                    vel_filt(1:en,:,kk) = repmat(vel_filt(en+1,:,kk),en,1,1) ;
                    loc(1:en,:,kk) = repmat(loc(en+1,:,kk),en,1,1) ;
                    loc_filt(1:en,:,kk) = repmat(loc_filt(en+1,:,kk),en,1,1) ;
                end
                
                if data_nan(end,:,kk) == 1
                    st = find(data_nan(end:-1:1,1,kk)==0,1) ;
                    vel(T-st+1:end,:,kk) = repmat(vel(T-st,:,kk),st-1,1,1) ;
                    vel_filt(T-st+1:end,:,kk) = repmat(vel_filt(T-st,:,kk),st-1,1,1) ;
                    loc(T-st+1:end,:,kk) = repmat(loc(T-st,:,kk),st-1,1,1) ;
                    loc_filt(T-st+1:end,:,kk) = repmat(loc_filt(T-st,:,kk),st-1,1,1) ;
                end
                
                if sum(isnan(loc(:,:,kk)))
                    disp('data includes NaN'); % error
                end
                
                kk = kk + 1 ;
            end
        end
        
        % redefine T
        T = find(all(data_nan,3),1,'first')-1;
        loc = loc(1:T,:,:);
        loc_filt = loc_filt(1:T,:,:);
        vel = vel(1:T,:,:);
        vel_filt = vel_filt(1:T,:,:);
        data_nan = data_nan(1:T,:,:);
        loc_nan = loc_nan(1:T,:,:);
        
        % pseudo label (not used)
        K = length(rawNo) ;
        label_time = zeros(T,K,K) ;
        for k = 1:K
            for j = 1:K
                for t = 1:T
                    if label(k) == label(j) && data_nan(t,1,k) == 0 && data_nan(t,1,j) == 0
                        label_time(t,k,j) = 1 ;
                    end
                end
            end
        end
        
        max_xy(f,:) = max(max(loc,[],1),[],3) ;
        min_xy(f,:) = min(min(loc,[],1),[],3) ;
        
        if 0 % figure
            figure(1)
            subplot 211
            plot(loc(:,1),'b') ; hold on ;
            plot(loc_filt(:,1),'r') ;
            subplot 212
            plot(vel(:,1),'b') ; hold on ;
            plot(vel_filt(:,1),'r') ;
        end
        % output
        dataset{f}.loc = loc_filt ;
        dataset{f}.vel = vel_filt ;
        dataset{f}.data_nan = data_nan ;
        
        dataset{f}.label_time = label_time ;
        dataset{f}.label = label ;
        dataset{f}.rawNo = rawNo ;
        dataset{f}.loc_raw = loc_filt ;
        dataset{f}.vel_raw = vel ;
        dataset{f}.followed = followed ;
        dataset{f}.K_max = K_max ;
        dataset{f}.Fs = Fs ;
        dataset{f}.max_xy = max_xy ;
        dataset{f}.min_xy = min_xy ;
        dataset{f}.loc_nan = loc_nan ;
        % save([mat_dir,filenames{f}],'loc','loc_filt','vel','vel_filt','followed','K_max') ;
    end
    save([mat_dir,'dataset_bats'],'dataset') ;
end

% figure (distance)
if 0
    for f = 1:length(filenames)
        figure(100+f)
        loc = dataset{f}.loc ;
        T = size(loc,1);
        K = size(loc,3);
        dist = NaN(T,K,K) ;
        for t= 1:T
            dist(t,:,:) = squareform(pdist(squeeze(loc(t,:,:))')) ;
        end
        data_nan = dataset{f}.data_nan ;
        for k = 1:K
            for j = 1:K
                if k < j
                    subplot(K,K,(k-1)*K+j)
                    plot(dist(:,k,j))
                    xlim([0 T])
                    ylim([0 20])
                end
            end
        end
        
    end
end

% movie (trajectories)
load([mat_dir,'dataset_bats']) ;
if 0
    for f = 1:length(filenames)
        close all
        n = 1 ;
        % load([mat_dir,filenames{f}]) ;
        pos = dataset{f}.loc_nan ;
        Start = 1 ; End = size(pos,1) ;
        Time = 1/Fs:1/Fs:End/Fs ;
        K = size(pos,3) ;
        
        % figure('visible','off'); % h = figure(1);
        set(gcf,'color',[1 1 1]) ;
        videoPath = [video_dir,filenames{f}]; % ['video_',num2str(n)] ;
        
        v = VideoWriter([videoPath,'.mp4'],'MPEG-4');
        open(v)
        
        nn = 1;
        clear mov
        duration = 30 ;
        for t = Start:End
            kk = 1 ;
            for k = 1:dataset{f}.K_max
                if ~isempty(find(k==dataset{f}.rawNo,1))% dataset{f}.followed(k) == 1
                    marker = 'o';
                    
                    xy = pos(t,:,kk) ;
                    if t <= duration
                        xy_long = pos(1:t,:,kk) ;
                    else; xy_long = pos(t-duration:t,:,kk) ;
                    end
                    ms = 12 ; lw = 1 ;
                    if mod(k,5) == 0 ; clr = 'r' ;
                    elseif mod(k,5) == 1; clr = 'g' ;
                    elseif mod(k,5) == 2; clr = 'b' ; %
                    elseif mod(k,5) == 3; clr = 'k' ;
                    elseif mod(k,5) == 4; clr = 'm' ;
                    end
                    
                    plot3(xy(1),xy(2),xy(3),marker,'markersize',ms,'linewidth',lw,'color',clr); hold on;
                    plot3(xy_long(:,1),xy_long(:,2),xy_long(:,3),'-','color',clr); hold on;
                    %text(xy(1),xy(2),xy(3),num2str(k));
                    text(xy(1),xy(2),xy(3),num2str(kk));
                    
                    kk = kk + 1 ;
                end
            end
            hold off
            
            axis equal
            set(gca,'xlim',[min_xy(f,1) max_xy(f,1)],'ylim',[min_xy(f,2) max_xy(f,2)],...
                'zlim',[min_xy(f,3) max_xy(f,3)],'View', [-10 14]) ; % [-84,12]);%
            xlabel('x')
            ylabel('y')
            zlabel('z')
            %axis off
            box off
            
            title([filenames{f},sprintf(', %d bats, Frame %d (%dHz)', K, t, floor(Fs))],'Fontsize',16) ; % Video %04d n,
            
            mov(nn)= getframe(gcf);  
            drawnow
            
            nn = nn+1;
            
        end
        writeVideo(v,mov)
        close(v)
    end
end

% figure (trajectories)
if 1
    for f = 1:length(filenames)
        % close all
        n = 1 ;
        % load([mat_dir,filenames{f}]) ;
        pos = dataset{f}.loc ;
        Start = 1 ; End = size(pos,1) ;
        Time = 1/Fs:1/Fs:End/Fs ;
        K = size(pos,3) ;
        
        h = figure(f*100);
        set(gcf,'color',[1 1 1]) ;
        
        
        nn = 1;
        clear mov
        duration = 30 ;
        kk = 1 ;
        for k = 1:dataset{f}.K_max
            if ~isempty(find(k==dataset{f}.rawNo,1))% dataset{f}.followed(k) == 1
                marker = 'o';
                
                xy_long = pos(1:End,:,kk) ;
                ms = 12 ; lw = 1 ;
                if mod(k,5) == 0 ; clr = 'r' ;
                elseif mod(k,5) == 1; clr = 'g' ;
                elseif mod(k,5) == 2; clr = 'b' ; %
                elseif mod(k,5) == 3; clr = 'k' ;
                elseif mod(k,5) == 4; clr = 'm' ;
                end
                
                plot3(xy_long(:,1),xy_long(:,2),xy_long(:,3),'-','color',clr,'linewidth',2); hold on;
                
                kk = kk + 1 ;
            end
            
            axis equal
            set(gca,'xlim',[min_xy(f,1) max_xy(f,1)],'ylim',[min_xy(f,2) max_xy(f,2)],...
                'zlim',[min_xy(f,3) max_xy(f,3)],'View', [-10 14]) ; % [-84,12]);%
            xlabel('x')
            ylabel('y')
            zlabel('z')
            %axis off
            box off
            
            title([filenames{f},sprintf(', K: %d, Frame %d (%dHz)', K, t, floor(Fs))],'Fontsize',16) ; % Video %04d n,
            
        end
        
    end
end


