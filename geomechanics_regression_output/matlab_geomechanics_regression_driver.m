try
addpath('C:/texnetwebtools/tools/fsp_python/reference_old_code/technical code');
addpath('C:/texnetwebtools/tools/fsp_python/reference_old_code/data entry functions');
hDV = struct();
hDV.data.stress.aphi.use = 0;
hDV.hfig = [];
sig = [5000,3500,4500];
pp0 = 2250;
strikes = [40;100;38;108;315];
dips = [85;84;78;67;66];
SHdir = 60;
mu = 0.6;
biot = 1;
nu = 0.5;
inputCell = {sig,0.00,pp0,strikes,dips,SHdir,0*strikes,mu,biot,nu};
[failout,outs,C1,C2,C3,sig_fault,tau_fault] = mohrs_3D(inputCell,hDV);
csvwrite('C:/texnetwebtools/tools/fsp_python/geomechanics_regression_output/matlab_fault_metrics.csv',real([failout(:),outs.cff(:),outs.scu(:),sig_fault(:),tau_fault(:)]));
csvwrite('C:/texnetwebtools/tools/fsp_python/geomechanics_regression_output/matlab_mohr_arcs.csv',[real(C1(1,:)).',imag(C1(1,:)).',real(C2(1,:)).',imag(C2(1,:)).',real(C3(1,:)).',imag(C3(1,:)).']);
samples = csvread('C:/texnetwebtools/tools/fsp_python/geomechanics_regression_output/mc_samples.csv'); mc_ppf = zeros(size(samples,1),1);

for sample_index = 1:size(samples,1)
    Sig0 = samples(sample_index,1:3)';
    p0_mc = samples(sample_index,4);
    strike_mc = samples(sample_index,5);
    dip_mc = samples(sample_index,6);
    SHdir_mc = samples(sample_index,7);
    mu_mc = samples(sample_index,8);
    inputCell = {Sig0,0.00,p0_mc,strike_mc,dip_mc,SHdir_mc,0,mu_mc,biot,nu};
    
    mc_ppf(sample_index) = mohrs_3D(inputCell,hDV);
end
csvwrite('C:/texnetwebtools/tools/fsp_python/geomechanics_regression_output/matlab_mc_slip_pressure.csv',mc_ppf);

catch ME
fid = fopen('C:/texnetwebtools/tools/fsp_python/geomechanics_regression_output/matlab_error.txt','w');
fprintf(fid,'%s\n',ME.message);
fclose(fid);
end
exit
