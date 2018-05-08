% EMD.M
%
% G. Rilling, last update: May 2005
%
% computes EMD (Empirical Mode Decomposition) according to:
%
% N. E. Huang et al., "The empirical mode decomposition and the 
% Hilbert spectrum for non-linear and non stationary time series analysis",  
% Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
%
% with variations reported in:
%
% G. Rilling, P. Flandrin and P. Gon�alv�s
% "On Empirical Mode Decomposition and its algorithms",
% IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing
% NSIP-03, Grado (I), June 2003
%
% default stopping criterion for sifting : 
%   at each point : mean amplitude < threshold2*envelope amplitude
%   &
%   mean of boolean array ((mean amplitude)/(envelope amplitude) > threshold) < tolerance
%   &
%   |#zeros-#extrema|<=1
%
% inputs:	
%		- x: analysed signal (line vector)
%		- opts (optional): struct object with (optional) fields:
%			- t: sampling times (line vector) (default: 1:length(x))
%			- stop: threshold, threshold2 and tolerance (optional)
%				for sifting stopping criterion 
%				default: [0.05,0.5,0.05]
%			- display: if equals to 1 shows sifting steps with pause
%				if equals to 2 shows sifting steps without pause (movie style)
%			- maxiterations: maximum number of sifting steps for the computation of each mode
%			- fix (int): disable the stopping criterion and do exactly
%				the value of the field number of sifting steps for each mode
%			- maxmodes: maximum number of imfs extracted
%			- interp: interpolation scheme: 'linear', 'cubic' or 'spline' (default)
%			- fix_h (int): do <fix_h> sifting iterations with |#zeros-#extrema|<=1 to stop 
%				according to N. E. Huang et al., "A confidence limit for the Empirical Mode 
%				Decomposition and Hilbert spectral analysis",
%				Proc. Royal Soc. London A, Vol. 459, pp. 2317-2345, 2003
%			- mask: masking signal used to improve the decomposition
%				according to R. Deering and J. F. Kaiser, "The use of a masking signal to 
%				improve empirical mode decomposition",
%				ICASSP 2005
%
% outputs: 
%		- imf: intrinsic mode functions (last line = residual)
%		- ort: index of orthogonality
%		- nbits: number of iterations for each mode
%
% calls:   
%		- io: computes the index of orthogonality
%
%examples:
%
%>>x = rand(1,512);
%
%>>imf = emd(x);
%
%>>imf = emd(x,struct('stop',[0.1,0.5,0.05],'maxiterations',100));
%Remark: the following syntax is equivalent
%>>imf = emd(x,'stop',[0.1,0.5,0.05],'maxiterations',100);
%
%>>options.dislpay = 1;
%>>options.fix = 10;
%>>options.maxmodes = 3;
%>>[imf,ort,nbits] = emd(x,options);



function [imf,ort,nbits] = emd(varargin);

[x,t,sd,sd2,tol,display_sifting,sdt,sd2t,ner,nzr,lx,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask] = init(varargin{:});	

if display_sifting
  figure
end

% maximum number of iterations
% MAXITERATIONS=2000;


%main loop : requires at least 3 extrema to proceed
while ~stop_EMD(r) & (k < MAXMODES+1 | MAXMODES == 0) & ~any(mask)

	% current mode
	m = r;
	

	% mode at previous iteration
	mp = m;

			
	if FIXE
		[stop_sift,moyenne] = stop_sifting_fixe(t,m,INTERP);
	elseif FIXE_H
		stop_count = 0;
		[stop_sift,moyenne,stop_count] = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H);
		stop_count = 0;
	else
		[stop_sift,moyenne] = stop_sifting(m,t,sd,sd2,tol,INTERP);
	end	

  	if (max(m) - min(m)) < (1e-10)*(max(x) - min(x))
    	if ~stop_sift
		   	warning('forced stop of EMD : too small amplitude')
    	else
      		disp('forced stop of EMD : too small amplitude')
    	end
    	break
  	end

    
  	% sifting loop
  	while ~stop_sift & nbit<MAXITERATIONS
    
    	if(nbit>MAXITERATIONS/5 & mod(nbit,floor(MAXITERATIONS/10))==0 & ~FIXE & nbit > 100)
      		disp(['mode ',int2str(k),', iteration ',int2str(nbit)])
			if exist('s')
	      		disp(['stop parameter mean value : ',num2str(s)])
			end
            [im,iM] = extr(m);
            disp([int2str(sum(m(im) > 0)),' minima > 0; ',int2str(sum(m(iM) < 0)),' maxima < 0.'])
		end
		
   		%sifting
    	m = m - moyenne;
		
		%computation of mean and stopping criterion  
		if FIXE
			[stop_sift,moyenne] = stop_sifting_fixe(t,m,INTERP);
		elseif FIXE_H
			[stop_sift,moyenne,stop_count] = stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H);
		else
			[stop_sift,moyenne,s] = stop_sifting(m,t,sd,sd2,tol,INTERP);  
		end		
	    
    	% display
        
    	if display_sifting
			[envminp,envmaxp,envmoyp] = envelope(t,mp,INTERP);
			if FIXE |FIXE_H
				display_emd_fixe(t,m,mp,r,envminp,envmaxp,envmoyp,nbit,k,display_sifting)
			else
			   	sxp=2*(abs(envmoyp))./(abs(envmaxp-envminp));
			   	sp = mean(sxp);
				display_emd(t,m,mp,r,envminp,envmaxp,envmoyp,s,sp,sxp,sdt,sd2t,nbit,k,display_sifting,stop_sift)
			end
    	end


    	mp = m;
    	nbit=nbit+1;
    	NbIt=NbIt+1;

    	if(nbit==(MAXITERATIONS-1) & ~FIXE & nbit > 100)
            if exist('s')
                warning(['forced stop of sifting : too many iterations... mode ',int2str(k),'. stop parameter mean value : ',num2str(s)])
            else
                warning(['forced stop of sifting : too many iterations... mode ',int2str(k),'.'])
            end
    	end
  
  	end % sifting loop
  	imf(k,:) = m;
  	if display_sifting
  		disp(['mode ',int2str(k),' stored'])
  	end
  	nbits(k) = nbit;
  	k = k+1;

	
  	r = r - m;
  	nbit=0;


end %main loop

if sum(r.^2) & ~any(mask)
	imf(k,:) = r;
end

ort = io(x,imf);

if display_sifting
  close
end

%---------------------------------------------------------------------------------------------------

function stop = stop_EMD(r)
	[indmin,indmax,indzer] = extr(r);
	ner = length(indmin) + length(indmax);
	stop = ner <3;	


%---------------------------------------------------------------------------------------------------


function [stop,envmoy,s]= stop_sifting(m,t,sd,sd2,tol,INTERP)
	try
	   	[envmin,envmax,envmoy,indmin,indmax,indzer] = envelope(t,m,INTERP);
	   	nem = length(indmin) + length(indmax);
	   	nzm = length(indzer);

	   	% evaluation of mean zero
	   	sx=2*(abs(envmoy))./(abs(envmax-envmin));
	   	s = mean(sx);

		stop = ~((mean(sx > sd) > tol | any(sx > sd2) | (abs(nzm-nem)>1)) & (nem > 2));
	catch
		disp(lasterr)
		stop = 1;
		envmoy = zeros(1,length(m));
% 		disp(['catch : ',lasterr])
		s = NaN;
	end


%---------------------------------------------------------------------------------------------------
function [stop,moyenne]= stop_sifting_fixe(t,m,INTERP)
	try
		[envmin,envmax,moyenne] = envelope(t,m,INTERP);
		stop = 0;
% 		disp('try ok')
	catch
		moyenne = zeros(1,length(m));
		stop = 1;
% 		disp(['catch : ',lasterr])
	end


%---------------------------------------------------------------------------------------------------
function [stop,moyenne,stop_count]= stop_sifting_fixe_h(t,m,INTERP,stop_count,FIXE_H)
	try
		[envmin,envmax,moyenne,indmin,indmax,indzer] = envelope(t,m,INTERP);
	   	nem = length(indmin) + length(indmax);
	   	nzm = length(indzer);

		if (abs(nzm-nem)>1)
			stop = 0;
			stop_count = 0;
		else
			stop_count = stop_count+1;
			stop = (stop_count == FIXE_H);
		end
% 		disp('try ok')
	catch
		moyenne = zeros(1,length(m));
		stop = 1;
% 		disp(['catch : ',lasterr])
	end


%---------------------------------------------------------------------------------------------------

function display_emd(t,m,mp,r,envmin,envmax,envmoy,s,sb,sx,sdt,sd2t,nbit,k,display_sifting,stop_sift)
	subplot(4,1,1)
    plot(t,mp);hold on;
    plot(t,envmax,'--k');plot(t,envmin,'--k');plot(t,envmoy,'r');
    title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' before sifting']);
    set(gca,'XTick',[])
    hold  off
    subplot(4,1,2)
    plot(t,sx)
    hold on
    plot(t,sdt,'--r')
    plot(t,sd2t,':k')
    title('stop parameter')
    set(gca,'XTick',[])
    hold off
    subplot(4,1,3)
    plot(t,m)
    title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' after sifting']);
    set(gca,'XTick',[])
    subplot(4,1,4);
    plot(t,r-m)
    title('residue');
    disp(['stop parameter mean value : ',num2str(sb),' before sifting and ',num2str(s),' after'])
	if stop_sift
		disp('last iteration for this mode')
	end
    if display_sifting == 2
      pause(0.01)
    else
      pause
    end

		       
%---------------------------------------------------------------------------------------------------

function display_emd_fixe(t,m,mp,r,envmin,envmax,envmoy,nbit,k,display_sifting)
	subplot(3,1,1)
    plot(t,mp);hold on;
    plot(t,envmax,'--k');plot(t,envmin,'--k');plot(t,envmoy,'r');
    title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' before sifting']);
    set(gca,'XTick',[])
    hold  off
    subplot(3,1,2)
    plot(t,m)
    title(['IMF ',int2str(k),';   iteration ',int2str(nbit),' after sifting']);
    set(gca,'XTick',[])
    subplot(3,1,3);
    plot(t,r-m)
    title('residue');
    if display_sifting == 2
      pause(0.01)
    else
      pause
  end
		       
%---------------------------------------------------------------------------------------------------

function [envmin, envmax,envmoy,indmin,indmax,indzer] = envelope(t,x,INTERP)
%computes envelopes and mean with various interpolations
	
NBSYM = 2;		
DEF_INTERP = 'spline';
	
	
if nargin < 2
	x = t;
	t = 1:length(x);
	INTERP = DEF_INTERP;
end

if nargin == 2
	if ischar(x)
		INTERP = x;
		x = t;
		t = 1:length(x);
	end
end

if ~ischar(INTERP)
	error('interp parameter must be ''linear'''', ''cubic'' or ''spline''')
end

if ~any(strcmpi(INTERP,{'linear','cubic','spline'}))
	error('interp parameter must be ''linear'''', ''cubic'' or ''spline''')
end

if min([size(x),size(t)]) > 1
	error('x and t must be vectors')
end
s = size(x);
if s(1) > 1
	x = x';
end
s = size(t);
if s(1) > 1
	t = t';
end
if length(t) ~= length(x)
	error('x and t must have the same length')
end

lx = length(x);
[indmin,indmax,indzer] = extr(x,t);
     

%boundary conditions for interpolation
		
[tmin,tmax,xmin,xmax] = boundary_conditions(indmin,indmax,t,x,NBSYM);

% definition of envelopes from interpolation

envmax = interp1(tmax,xmax,t,INTERP);	
envmin = interp1(tmin,xmin,t,INTERP);

if nargout > 2
    envmoy = (envmax + envmin)/2;
end


%---------------------------------------------------------------------------------------

function [tmin,tmax,xmin,xmax] = boundary_conditions(indmin,indmax,t,x,nbsym)
% computes the boundary conditions for interpolation (mainly mirror symmetry)

	
	lx = length(x);
	
	if (length(indmin) + length(indmax) < 3)
% 		error('not enough extrema')
	end

	if indmax(1) < indmin(1)
    	if x(1) > x(indmin(1))
			lmax = fliplr(indmax(2:min(end,nbsym+1)));
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = indmax(1);
		else
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = [fliplr(indmin(1:min(end,nbsym-1))),1];
			lsym = 1;
		end
	else

		if x(1) < x(indmax(1))
			lmax = fliplr(indmax(1:min(end,nbsym)));
			lmin = fliplr(indmin(2:min(end,nbsym+1)));
			lsym = indmin(1);
		else
			lmax = [fliplr(indmax(1:min(end,nbsym-1))),1];
			lmin = fliplr(indmin(1:min(end,nbsym)));
			lsym = 1;
		end
	end
    
	if indmax(end) < indmin(end)
		if x(end) < x(indmax(end))
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = fliplr(indmin(max(end-nbsym,1):end-1));
			rsym = indmin(end);
		else
			rmax = [lx,fliplr(indmax(max(end-nbsym+2,1):end))];
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = lx;
		end
	else
		if x(end) > x(indmin(end))
			rmax = fliplr(indmax(max(end-nbsym,1):end-1));
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
			rsym = indmax(end);
		else
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
			rmin = [lx,fliplr(indmin(max(end-nbsym+2,1):end))];
			rsym = lx;
		end
	end
    
	tlmin = 2*t(lsym)-t(lmin);
	tlmax = 2*t(lsym)-t(lmax);
	trmin = 2*t(rsym)-t(rmin);
	trmax = 2*t(rsym)-t(rmax);
    
	% in case symmetrized parts do not extend enough
	if tlmin(1) > t(1) | tlmax(1) > t(1)
		if lsym == indmax(1)
			lmax = fliplr(indmax(1:min(end,nbsym)));
		else
			lmin = fliplr(indmin(1:min(end,nbsym)));
		end
		if lsym == 1
			error('bug')
		end
		lsym = 1;
		tlmin = 2*t(lsym)-t(lmin);
		tlmax = 2*t(lsym)-t(lmax);
	end   
    
	if trmin(end) < t(lx) | trmax(end) < t(lx)
		if rsym == indmax(end)
			rmax = fliplr(indmax(max(end-nbsym+1,1):end));
		else
			rmin = fliplr(indmin(max(end-nbsym+1,1):end));
		end
	if rsym == lx
		error('bug')
	end
		rsym = lx;
		trmin = 2*t(rsym)-t(rmin);
		trmax = 2*t(rsym)-t(rmax);
	end 
          
	xlmax =x(lmax); 
	xlmin =x(lmin);
	xrmax =x(rmax); 
	xrmin =x(rmin);
     
	tmin = [tlmin t(indmin) trmin];
	tmax = [tlmax t(indmax) trmax];
	xmin = [xlmin x(indmin) xrmin];
	xmax = [xlmax x(indmax) xrmax];
  
%---------------------------------------------------------------------------------------------------

function [indmin, indmax, indzer] = extr(x,t);
%extracts the indices corresponding to extrema

if(nargin==1)
  t=1:length(x);
end

m = length(x);

if nargout > 2
	x1=x(1:m-1);
	x2=x(2:m);
	indzer = find(x1.*x2<0);
	
	if any(x == 0)
	  iz = find( x==0 );
	  indz = [];
	  if any(diff(iz)==1)
	    zer = x == 0;
	    dz = diff([0 zer 0]);
	    debz = find(dz == 1);
	    finz = find(dz == -1)-1;
	    indz = round((debz+finz)/2);
	  else
	    indz = iz;
	  end
	  indzer = sort([indzer indz]);
	end
end
  
d = diff(x);

n = length(d);
d1 = d(1:n-1);
d2 = d(2:n);
indmin = find(d1.*d2<0 & d1<0)+1;
indmax = find(d1.*d2<0 & d1>0)+1;


% when two or more consecutive points have the same value we consider only one extremum in the middle of the constant area

if any(d==0)
  
  imax = [];
  imin = [];
  
  bad = (d==0);
  dd = diff([0 bad 0]);
  debs = find(dd == 1);
  fins = find(dd == -1);
  if debs(1) == 1
    if length(debs) > 1
      debs = debs(2:end);
      fins = fins(2:end);
    else
      debs = [];
      fins = [];
    end
  end
  if length(debs) > 0
    if fins(end) == m
      if length(debs) > 1
        debs = debs(1:(end-1));
        fins = fins(1:(end-1));

      else
        debs = [];
        fins = [];
      end      
    end
  end
  lc = length(debs);
  if lc > 0
    for k = 1:lc
      if d(debs(k)-1) > 0
        if d(fins(k)) < 0
          imax = [imax round((fins(k)+debs(k))/2)];
        end
      else
        if d(fins(k)) > 0
          imin = [imin round((fins(k)+debs(k))/2)];
        end
      end
    end
  end
  
  if length(imax) > 0
    indmax = sort([indmax imax]);
  end

  if length(imin) > 0
    indmin = sort([indmin imin]);
  end
  
end  
  
%---------------------------------------------------------------------------------------------------

function [x,t,sd,sd2,tol,display_sifting,sdt,sd2t,ner,nzr,lx,r,imf,k,nbit,NbIt,MAXITERATIONS,FIXE,FIXE_H,MAXMODES,INTERP,mask] = init(varargin)	

x = varargin{1};
if nargin == 2
	if strcmp(class(varargin{2}),'struct') 
		inopts = varargin{2};
	else
		error('when using 2 arguments the first one is the analysed signal x and the second one is a struct object describing the options')
	end
elseif nargin > 2
	try
		inopts = struct(varargin{2:end});
	catch
		error('bad argument syntax')
	end
end

% default for stopping
defstop = [0.05,0.5,0.05];

opt_fields = {'t','stop','display','maxiterations','fix','maxmodes','interp','fix_h','mask'};

defopts.stop = defstop;
defopts.display = 0;
defopts.t = 1:max(size(x));
defopts.maxiterations = 2000;
defopts.fix = 0;
defopts.maxmodes = 0;
defopts.interp = 'spline';
defopts.fix_h = 0;
defopts.mask = 0;

opts = defopts;



if(nargin==1)
	inopts = defopts;
elseif nargin == 0
	error('not enough arguments')
end


names = fieldnames(inopts);
for nom = names'
	if length(strmatch(char(nom), opt_fields)) == 0
		error(['bad option field name: ',char(nom)])
	end
	eval(['opts.',char(nom),' = inopts.',char(nom),';'])
end

t = opts.t;
stop = opts.stop;
display_sifting = opts.display;
MAXITERATIONS = opts.maxiterations;
FIXE = opts.fix;
MAXMODES = opts.maxmodes;
INTERP = opts.interp;
FIXE_H = opts.fix_h;
mask = opts.mask;

S = size(x);
if ((S(1) > 1) & (S(2) > 1)) | (length(S) > 2)
  error('x must have only one row or one column')
end

if S(1) > 1
  x = x';
end

S = size(t);
if ((S(1) > 1) & (S(2) > 1)) | (length(S) > 2)
  error('option field t must have only one row or one column')
end

if S(1) > 1
  t = t';
end

if (length(t)~=length(x))
  error('x and option field t must have the same length')
end

S = size(stop);
if ((S(1) > 1) & (S(2) > 1)) | (S(1) > 3) | (S(2) > 3) | (length(S) > 2)
  error('option field stop must have only one row or one column of max three elements')
end

if ~all(isfinite(x))
	error('data elements must be finite')
end

if S(1) > 1
  stop = stop';
  S = size(stop);
end

if S(2) < 3
  stop(3)=defstop(3);
end

if S(2) < 2
  stop(2)=defstop(2);
end


if ~ischar(INTERP)
	error('interp field must be ''linear'', ''cubic'' or ''spline''')
end

if ~any(strcmpi(INTERP,{'linear','cubic','spline'}))
	error('interp field must be ''linear'', ''cubic'' or ''spline''')
end

%special procedure when a masking signal is specified
if any(mask)
	S = size(mask);
	if min(S) > 1
		error('masking signal must have the same dimension as the analyzed signal x')
	end
	if S(1) > 1
		mask = mask';
	end
	if max(S) ~= max(size(x))
		error('masking signal must have the same dimension as the analyzed signal x')
	end
	opts.mask = 0;
	imf1 = emd(x+mask,opts);
	imf2 = emd(x-mask,opts);
	if size(imf1,1) ~= size(imf2,1)
		warning(['the two sets of IMFs have different sizes: ',int2str(size(imf1,1)),' and ',int2str(size(imf2,1)),' IMFs.'])
	end
	S1 = size(imf1,1);
	S2 = size(imf2,1);
	if S1 ~= S2
		if S1 < S2
			tmp = imf1;
			imf1 = imf2;
			imf2 = tmp;
		end
		imf2(max(S1,S2),1) = 0;
	end
	imf = (imf1+imf2)/2;

end


sd = stop(1);
sd2 = stop(2);
tol = stop(3);

lx = length(x);

sdt = sd*ones(1,lx);
sd2t = sd2*ones(1,lx);

if FIXE
	MAXITERATIONS = FIXE;
	if FIXE_H
		error('cannot use both ''fix'' and ''fix_h'' modes') 
	end
end

% number of extrema and zero-crossings in residual
ner = lx;
nzr = lx;

r = x;

if ~any(mask) % if a masking signal is specified "imf" already exists at this stage
	imf = [];
end
k = 1;

% iterations counter for extraction of 1 mode
nbit=0;

% total iterations counter
NbIt=0;

%---------------------------------------------------------------------------------------------------
