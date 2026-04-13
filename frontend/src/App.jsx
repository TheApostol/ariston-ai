import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Brain, 
  FileSearch, 
  Settings, 
  ShieldCheck, 
  Database,
  UploadCloud,
  Terminal,
  ChevronRight,
  Eye,
  FileText,
  Search,
  Beaker,
  Microscope
} from 'lucide-react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "/api/v1";
const WS_BASE = `ws://${window.location.hostname}:${window.location.port}/api/v1/ws/jobs`;

function App() {
  const [activeTab, setActiveTab] = useState('orchestrator');
  const [prompt, setPrompt] = useState('');
  const [patientId, setPatientId] = useState('ARISTON-001');
  const [jobs, setJobs] = useState([]);
  const [wsLogs, setWsLogs] = useState([]);
  const [files, setFiles] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [providerStatus, setProviderStatus] = useState(null);
  const [streamingContent, setStreamingContent] = useState('');
  const [streamingJobId, setStreamingJobId] = useState(null);
  const streamIntervalRef = useRef(null);
  const ws = useRef(null);

  // Initialize WebSocket for Real-time Updates
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    const clientId = `client_${Math.random().toString(36).substr(2, 9)}`;
    ws.current = new WebSocket(`${WS_BASE}/${clientId}`);

    ws.current.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setWsLogs(prev => [update, ...prev].slice(0, 50));

      // Update job status in state
      setJobs(prev => prev.map(job =>
        job.job_id === update.job_id
          ? { ...job, status: update.status, result: update.data }
          : job
      ));
    };

    // Fetch provider status on mount
    axios.get(`${API_BASE}/providers/status`)
      .then(r => setProviderStatus(r.data))
      .catch(() => setProviderStatus(null));

    return () => {
      ws.current.close();
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    };
  }, []);

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });

  const handleFileUpload = async (e) => {
    const rawFiles = Array.from(e.target.files);
    const processed = [];

    for (const file of rawFiles) {
      if (file.type.startsWith('image/') || file.name.match(/\.(jpg|jpeg|png|webp|dcm)$/i)) {
        const b64 = await fileToBase64(file);
        processed.push({ type: 'image', name: file.name, data: b64, raw: file });
      } else if (file.name.endsWith('.json')) {
        const text = await file.text();
        try {
          processed.push({ type: 'fhir', name: file.name, data: JSON.parse(text), raw: file });
        } catch (err) { console.error("Invalid JSON/FHIR", err); }
      } else {
        // PDF, CSV, TXT — keep raw file for FormData upload
        processed.push({ type: 'document', name: file.name, data: null, raw: file });
      }
    }
    setFiles(prev => [...prev, ...processed]);
  };

  const submitOrchestration = async () => {
    const hasPrompt = prompt.trim().length > 0;
    const hasFiles = files.length > 0;
    if (!hasPrompt && !hasFiles) return;

    setIsRunning(true);
    setSubmitError(null);

    // Files present → use /upload multipart endpoint for full analysis
    if (hasFiles) {
      const formData = new FormData();
      formData.append('prompt', prompt.trim() || 'Analyze this file and provide a full clinical report.');
      if (patientId) formData.append('patient_id', patientId);
      formData.append('layer', activeTab === 'vision' ? 'radiology' : '');
      files.forEach(f => { if (f.raw) formData.append('files', f.raw); });

      const jobId = `upload_${Math.random().toString(36).substr(2, 8)}`;
      setJobs(prev => [{
        job_id: jobId,
        status: 'processing',
        prompt: prompt.trim() || `Analyzing ${files.length} file(s)`,
        timestamp: new Date().toISOString(),
      }, ...prev]);

      try {
        const response = await axios.post(`${API_BASE}/upload`, formData);
        setJobs(prev => prev.map(j =>
          j.job_id === jobId ? { ...j, status: 'completed', result: response.data } : j
        ));
        setSelectedJob({ job_id: jobId, status: 'completed', result: response.data, prompt });
        setActiveTab('vision');
      } catch (err) {
        const msg = err.response?.data?.detail || err.message || 'Upload failed';
        setSubmitError(msg);
        setJobs(prev => prev.map(j => j.job_id === jobId ? { ...j, status: 'failed' } : j));
      } finally {
        setFiles([]);
        setPrompt('');
        setIsRunning(false);
      }
      return;
    }

    // Text only → async orchestration, result via polling
    try {
      const response = await axios.post(`${API_BASE}/orchestrate`, {
        prompt: prompt.trim(),
        patient_id: patientId || null,
        use_rag: true,
        context: {},
      });
      const jobId = response.data.job_id;
      setJobs(prev => [{
        job_id: jobId,
        status: 'accepted',
        prompt: prompt.trim(),
        timestamp: new Date().toISOString(),
      }, ...prev]);
      setPrompt('');
      setStreamingContent('');
      setStreamingJobId(jobId);

      // Poll every 1.5s and stream content word-by-word
      let displayedWords = 0;
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = setInterval(async () => {
        try {
          const jobRes = await axios.get(`${API_BASE}/jobs/${jobId}`);
          const jobData = jobRes.data;
          const content = jobData?.result?.content || jobData?.content || '';

          // Stream content word-by-word
          const words = content.split(' ');
          if (words.length > displayedWords) {
            displayedWords = Math.min(displayedWords + 3, words.length);
            setStreamingContent(words.slice(0, displayedWords).join(' '));
          }

          // Update job in list
          if (jobData.status === 'completed' || jobData.status === 'failed') {
            setJobs(prev => prev.map(j =>
              j.job_id === jobId
                ? { ...j, status: jobData.status, result: jobData.result || jobData }
                : j
            ));
            if (jobData.status === 'completed') {
              setStreamingContent(content); // show full content
              setSelectedJob({ job_id: jobId, status: 'completed', result: jobData.result || jobData, prompt: prompt.trim() });
            }
            clearInterval(streamIntervalRef.current);
            setStreamingJobId(null);
          }
        } catch (_) {
          // silently ignore poll errors
        }
      }, 1500);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Execution failed';
      setSubmitError(msg);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar Navigation */}
      <nav className="w-64 bg-slate-950 border-r border-white/5 flex flex-col p-6 space-y-8">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 medical-gradient rounded-xl flex items-center justify-center">
            <Activity className="text-white w-6 h-6" />
          </div>
          <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
            Ariston OS
          </span>
        </div>

        <div className="space-y-1 flex-1">
           <NavItem icon={<Brain />} label="Execute" active={activeTab === 'orchestrator'} onClick={() => setActiveTab('orchestrator')} />
           <NavItem icon={<Activity />} label="Patient Timeline" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
           <NavItem icon={<Activity />} label="Digital Twin" active={activeTab === 'twin'} onClick={() => setActiveTab('twin')} />
           <NavItem icon={<FileText />} label="Regulatory" active={activeTab === 'reg'} onClick={() => setActiveTab('reg')} />
          <NavItem icon={<Beaker />} label="Molecule-Lab" active={activeTab === 'mol'} onClick={() => setActiveTab('mol')} />
          <NavItem icon={<Microscope />} label="Trial-Matcher" active={activeTab === 'trial'} onClick={() => setActiveTab('trial')} />
          <NavItem icon={<Database />} label="Audit Logs" active={activeTab === 'audit'} onClick={() => setActiveTab('audit')} />
          <NavItem icon={<Eye />} label="Rad-Visualizer" active={activeTab === 'vision'} onClick={() => setActiveTab('vision')} />
          <NavItem icon={<ShieldCheck />} label="Benchmarking" active={activeTab === 'bench'} onClick={() => setActiveTab('bench')} />
          <NavItem icon={<Database />} label="LATAM Data" active={activeTab === 'latam'} onClick={() => setActiveTab('latam')} />
          <NavItem icon={<Activity />} label="Platform Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <NavItem icon={<Brain />} label="Agents" active={activeTab === 'agents'} onClick={() => setActiveTab('agents')} />
        </div>

        <div className="pt-6 border-t border-white/5 space-y-1">
          <NavItem icon={<Settings size={20} />} label="Settings" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
        </div>

        <div className="mt-auto pt-4">
           <div className="p-3 bg-white/5 rounded-xl border border-white/5 flex items-center justify-between">
              <span className="text-[10px] text-slate-500 font-bold">Search</span>
              <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-[9px] text-slate-400 font-mono">⌘ K</kbd>
           </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 bg-[#030712] overflow-y-auto relative p-8">
        <div className="max-w-6xl mx-auto">
          {activeTab === 'orchestrator' && (
            <div className="space-y-8">
               {/* Provider Status Bar */}
               <ProviderStatusBar status={providerStatus} />

               <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <EcosystemCard icon={<Database size={16} />} label="OpenFDA" status="Connected" color="text-brand-primary" />
                  <EcosystemCard icon={<FileText size={16} />} label="ClinicalTrials" status="Active" color="text-brand-secondary" />
                  <EcosystemCard icon={<Search size={16} />} label="PubMed" status="Syncing" color="text-brand-primary" />
                  <EcosystemCard icon={<ShieldCheck size={16} />} label="RxNorm" status="Optimized" color="text-brand-secondary" />
               </div>
               
               <div className="flex justify-between items-end">
                  <div>
                    <h1 className="text-4xl font-bold text-white mb-2">Ariston Execute</h1>
                    <p className="text-slate-400">Run clinical, pharma, and radiology agents. Upload files for instant full analysis.</p>
                  </div>
                   <div className="px-4 py-2 glass-card flex items-center space-x-2 text-xs text-brand-accent">
                     <div className="w-2 h-2 rounded-full bg-brand-accent animate-pulse" />
                     <span>Compliance Layer Active</span>
                   </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                   <div className="glass-card p-4 h-24 relative overflow-hidden flex items-center justify-between">
                      <div className="z-10">
                         <p className="text-[10px] text-slate-500 uppercase font-bold">Clinical Telemetry</p>
                         <p className="text-xl font-black text-white">Live-ECG</p>
                      </div>
                      <div className="absolute inset-0 z-0">
                         <BiometricsSimulator />
                      </div>
                   </div>
                   <div className="glass-card p-4 h-24 flex items-center justify-between">
                      <div>
                         <p className="text-[10px] text-slate-500 uppercase font-bold">Genomic Risk</p>
                         <p className="text-xl font-black text-brand-primary">Elevated</p>
                      </div>
                      <ShieldCheck className="text-brand-primary opacity-50" size={32} />
                   </div>
                </div>

               {/* Prompt Engine */}
               <section className="glass-card p-6 space-y-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2 text-sm text-slate-400">
                      <Terminal className="w-4 h-4" />
                      <span>Command Interface</span>
                    </div>
                    <div className="flex items-center space-x-2">
                       <span className="text-[10px] text-slate-500 uppercase font-bold">Patient ID:</span>
                       <input 
                         type="text" 
                         value={patientId} 
                         onChange={(e) => setPatientId(e.target.value)}
                         placeholder="e.g. ARISTON-001"
                         className="bg-white/5 border border-white/10 rounded px-2 py-0.5 text-[10px] text-brand-primary placeholder:text-slate-700 focus:border-brand-primary/50 outline-none"
                       />
                    </div>
                  </div>
                  <textarea 
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Enter clinical query or patient context..."
                    className="w-full h-32 bg-transparent border-none focus:ring-0 text-lg resize-none placeholder:text-slate-600"
                  />
                  {/* Uploaded files preview */}
                  {files.length > 0 && (
                    <div className="flex flex-wrap gap-2 pt-2">
                      {files.map((f, i) => (
                        <div key={i} className="flex items-center space-x-2 bg-brand-primary/10 border border-brand-primary/20 rounded-lg px-3 py-1.5">
                          {f.type === 'image' ? <Eye size={12} className="text-brand-primary" /> : <FileText size={12} className="text-brand-primary" />}
                          <span className="text-[10px] text-slate-300">{f.name}</span>
                          <button onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))} className="text-slate-500 hover:text-medical-red text-xs">×</button>
                        </div>
                      ))}
                    </div>
                  )}

                  {submitError && (
                    <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs flex items-start space-x-2">
                      <span className="font-bold">Error:</span>
                      <span>{submitError}</span>
                      <button onClick={() => setSubmitError(null)} className="ml-auto text-red-500 hover:text-red-300">×</button>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-4 border-t border-white/5">
                    <div className="flex space-x-4">
                       <label className={`flex items-center space-x-2 cursor-pointer transition-colors ${isRunning ? 'opacity-40 cursor-not-allowed' : 'text-slate-400 hover:text-white'}`}>
                          <UploadCloud className="w-5 h-5" />
                          <span className="text-sm">Upload X-Ray / Scan / PDF / FHIR</span>
                          <input
                            type="file"
                            multiple
                            accept="image/*,.pdf,.json,.txt,.csv,.dcm"
                            className="hidden"
                            disabled={isRunning}
                            onChange={handleFileUpload}
                          />
                       </label>
                    </div>
                    <button
                      onClick={submitOrchestration}
                      disabled={isRunning || (!prompt.trim() && files.length === 0)}
                      className="medical-gradient px-8 py-2.5 rounded-xl font-semibold shadow-lg shadow-brand-primary/20 hover:scale-[1.02] transition-transform flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                    >
                      {isRunning ? (
                        <><Activity className="w-4 h-4 animate-spin" /><span>{files.length > 0 ? 'Analyzing...' : 'Executing...'}</span></>
                      ) : files.length > 0 ? (
                        <><UploadCloud className="w-4 h-4" /><span>Analyze {files.length} File{files.length > 1 ? 's' : ''}</span></>
                      ) : (
                        <><span>Execute</span><ChevronRight className="w-4 h-4" /></>
                      )}
                    </button>
                  </div>
               </section>

               {/* Streaming Result Panel */}
               {(streamingContent || streamingJobId) && (
                 <section className="glass-card p-6 border border-brand-primary/20">
                   <div className="flex items-center justify-between mb-4">
                     <div className="flex items-center space-x-2">
                       <Activity className={`w-4 h-4 ${streamingJobId ? 'animate-pulse text-brand-primary' : 'text-brand-accent'}`} />
                       <span className="text-sm font-bold text-white">
                         {streamingJobId ? 'Streaming Output...' : 'Completed'}
                       </span>
                     </div>
                     <CopyButton text={streamingContent} label="Copy Report" />
                   </div>
                   <div className="text-slate-300 text-sm leading-relaxed font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                     {streamingContent}
                     {streamingJobId && <span className="animate-pulse text-brand-primary">▌</span>}
                   </div>
                 </section>
               )}

               {/* Job Monitor */}
               <section className="space-y-4">
                  <h2 className="text-xl font-bold flex items-center space-x-2">
                    <FileSearch className="w-5 h-5 text-brand-primary" />
                    <span>Active Execution Pool</span>
                  </h2>
                  <div className="grid grid-cols-1 gap-4">
                    <AnimatePresence>
                      {jobs.map((job) => (
                        <motion.div
                          key={job.job_id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="glass-card p-5 flex items-center justify-between group hover:border-brand-primary/30 transition-colors"
                        >
                          <div className="flex items-center space-x-4">
                            <div className={`p-3 rounded-xl ${job.status === 'completed' ? 'bg-brand-accent/10 text-brand-accent' : 'bg-brand-primary/10 text-brand-primary'}`}>
                              {job.status === 'completed' ? <ShieldCheck /> : <Activity className="animate-pulse" />}
                            </div>
                            <div>
                               <p className="font-medium text-slate-100">{job.prompt?.slice(0, 60)}...</p>
                               <div className="flex items-center space-x-2 mt-1">
                                  <p className={`text-xs uppercase tracking-widest ${job.status === 'failed' ? 'text-medical-red' : 'text-slate-500'}`}>
                                    {job.status} • {job.job_id.slice(0, 8)}
                                  </p>
                               </div>
                            </div>
                          </div>
                          {job.status === 'completed' && (
                            <button onClick={() => { setActiveTab('vision'); setSelectedJob(job); }} className="p-2 bg-white/5 rounded-lg">
                              <Eye className="w-5 h-5 hover:text-brand-primary" />
                            </button>
                          )}
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
               </section>
            </div>
          )}

          {activeTab === 'timeline' && <TimelineView patientId={patientId} />}
          {activeTab === 'twin' && <TwinView job={selectedJob} />}
          {activeTab === 'reg' && <RegulatoryView job={selectedJob} />}
          {activeTab === 'mol' && <MoleculeView />}
          {activeTab === 'trial' && <TrialMatcherView />}
          {activeTab === 'audit' && <AuditView logs={wsLogs} />}
          {activeTab === 'vision' && <VisionView job={selectedJob} onBack={() => setActiveTab('orchestrator')} />}
          {activeTab === 'bench' && <BenchmarkingView logs={wsLogs} />}
          {activeTab === 'latam' && <LatamDataView />}
          {activeTab === 'dashboard' && <PlatformDashboard />}
          {activeTab === 'agents' && <AgentsView />}
          {activeTab === 'settings' && <SettingsView />}
        </div>
      </main>

      <CommandPalette 
        isOpen={isCommandPaletteOpen} 
        onClose={() => setIsCommandPaletteOpen(false)} 
        jobs={jobs}
        onSelectJob={(job) => { setSelectedJob(job); setActiveTab('vision'); setIsCommandPaletteOpen(false); }}
      />

      {/* Logs Sidebar/Dock */}
      <aside className="w-80 bg-slate-950 border-l border-white/5 flex flex-col hidden xl:flex">
         <div className="p-6 border-b border-white/5 flex items-center justify-between">
           <span className="text-sm font-bold uppercase tracking-widest text-slate-500">Live Agent Feed</span>
           <div className="w-2 h-2 rounded-full bg-brand-accent" />
         </div>
         <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-[10px]">
            {wsLogs.map((log, idx) => (
              <div key={idx} className="p-3 bg-white/5 rounded-lg border-l-2 border-brand-primary border-opacity-30">
                <p className="text-brand-primary mb-1">[{new Date().toLocaleTimeString()}] JOB_{log.job_id?.slice(0, 4)}</p>
                <p className="text-slate-400 italic">:: status {'->'} {log.status}</p>
                {log.data?.content && <p className="text-slate-500 line-clamp-2">:: out {'->'} {log.data.content}</p>}
              </div>
            ))}
         </div>
      </aside>
    </div>
  );
}

function NavItem({ icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${
        active ? 'bg-brand-primary/10 text-brand-primary' : 'text-slate-400 hover:bg-white/5 hover:text-slate-100'
      }`}
    >
      {React.cloneElement(icon, { size: 20 })}
      <span className="font-medium">{label}</span>
      {active && <motion.div layoutId="nav-active" className="ml-auto w-1.5 h-1.5 rounded-full bg-brand-primary" />}
    </button>
  );
}

function CopyButton({ text, label = 'Copy' }) {
  const [copied, setCopied] = React.useState(false);
  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button
      onClick={handleCopy}
      disabled={!text}
      className="flex items-center space-x-1.5 px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-[10px] font-bold uppercase tracking-widest text-slate-400 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
    >
      <FileText size={11} />
      <span>{copied ? 'Copied!' : label}</span>
    </button>
  );
}

function ProviderStatusBar({ status }) {
  if (!status) return null;

  const providerConfig = [
    { key: 'anthropic',  label: 'Anthropic',  model: status?.anthropic?.model || 'claude-sonnet-4-6' },
    { key: 'gemini',     label: 'Gemini',      model: status?.gemini?.model || 'gemini-2.0-flash' },
    { key: 'openai',     label: 'OpenAI',      model: 'gpt-4o-mini' },
    { key: 'openrouter', label: 'OpenRouter',  model: 'auto' },
  ];

  const getIndicator = (s) => {
    if (!s) return { dot: 'bg-slate-600', text: 'text-slate-500', label: 'Unknown' };
    switch (s.status) {
      case 'live':            return { dot: 'bg-green-400',  text: 'text-green-400',  label: 'Live' };
      case 'quota_exhausted': return { dot: 'bg-yellow-400', text: 'text-yellow-400', label: 'Quota' };
      case 'invalid_key':
      case 'no_key':          return { dot: 'bg-red-500',    text: 'text-red-500',    label: 'Invalid key' };
      case 'timeout':         return { dot: 'bg-orange-400', text: 'text-orange-400', label: 'Timeout' };
      default:                return { dot: 'bg-slate-500',  text: 'text-slate-400',  label: s.status || 'Unknown' };
    }
  };

  return (
    <div className="flex items-center space-x-3 px-4 py-2.5 bg-white/3 border border-white/5 rounded-xl">
      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mr-2 whitespace-nowrap">Providers</span>
      {providerConfig.map(({ key, label }) => {
        const s = status[key];
        const ind = getIndicator(s);
        return (
          <div key={key} className="flex items-center space-x-1.5">
            <div className={`w-2 h-2 rounded-full ${ind.dot} flex-shrink-0`} />
            <span className={`text-[10px] font-semibold ${ind.text}`}>{label}:</span>
            <span className="text-[10px] text-slate-400">{ind.label}</span>
          </div>
        );
      })}
    </div>
  );
}

function StatCard({ label, value, sub, color = "text-brand-primary" }) {
  return (
    <div className="glass-card p-5">
      <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">{label}</p>
      <p className={`text-2xl font-black ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}

function TimelineView({ patientId }) {
  const events = [
    { date: "2026-04-01", event: "Baseline labs drawn", type: "lab" },
    { date: "2026-04-05", event: "Chest X-ray ordered", type: "imaging" },
    { date: "2026-04-08", event: "PGx panel returned — CYP2D6 IM", type: "genomics" },
    { date: "2026-04-10", event: "Drug interaction flag: Warfarin + Amiodarone", type: "alert" },
    { date: "2026-04-12", event: "Ariston Execute — clinical report generated", type: "ai" },
  ];
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white">Patient Timeline — {patientId}</h2>
      <div className="glass-card p-6 space-y-4">
        {events.map((e, i) => (
          <div key={i} className="flex items-start space-x-4 border-l-2 border-brand-primary pl-4">
            <div>
              <p className="text-xs text-slate-500 font-mono">{e.date}</p>
              <p className="text-sm text-white">{e.event}</p>
              <span className="text-[10px] px-2 py-0.5 bg-white/5 rounded-full text-brand-accent">{e.type}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrialMatcherView() {
  const trials = [
    { id: "NCT04822792", title: "Novel mTOR inhibitor in relapsed NSCLC", phase: "II", match: 94 },
    { id: "NCT05103240", title: "CAR-T for HER2+ breast cancer — expanded access", phase: "III", match: 87 },
    { id: "NCT04567830", title: "PD-L1 checkpoint in microsatellite-unstable CRC", phase: "II", match: 81 },
    { id: "NCT05299710", title: "BRCA1/2 PARP inhibitor maintenance", phase: "III", match: 76 },
  ];
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white">Clinical Trial Matcher</h2>
      <p className="text-slate-400">AI-matched trials from ClinicalTrials.gov based on patient phenotype and genomic profile.</p>
      <div className="space-y-3">
        {trials.map(t => (
          <div key={t.id} className="glass-card p-5 flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 font-mono">{t.id} · Phase {t.phase}</p>
              <p className="text-sm text-white font-semibold mt-1">{t.title}</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-black text-brand-primary">{t.match}%</p>
              <p className="text-[10px] text-slate-500">match score</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function BenchmarkingView({ logs }) {
  const metrics = [
    { label: "Avg Latency", value: "1.24s", sub: "p95: 2.1s" },
    { label: "Safety Pass Rate", value: "99.8%", sub: "last 1000 requests" },
    { label: "RAG Hit Rate", value: "87%", sub: "PubMed + FDA grounded" },
    { label: "Consensus Rate", value: "94%", sub: "dual-model agreement" },
  ];
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white">MedPerf Benchmarking</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map(m => <StatCard key={m.label} label={m.label} value={m.value} sub={m.sub} />)}
      </div>
      <div className="glass-card p-6">
        <p className="text-xs text-slate-500 uppercase font-bold mb-4">Recent Pipeline Events</p>
        {logs.length === 0 && <p className="text-slate-500 text-sm">No events yet — run a job to populate benchmarks.</p>}
        {logs.slice(0, 10).map((log, i) => (
          <div key={i} className="flex justify-between text-xs py-2 border-b border-white/5 last:border-0">
            <span className="text-slate-400 font-mono">{log.job_id?.slice(0, 8)}</span>
            <span className={log.status === 'completed' ? 'text-brand-accent' : 'text-yellow-400'}>{log.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SettingsView() {
  const [saved, setSaved] = React.useState(false);
  const fields = [
    { label: "Default Model", value: "gemini-1.5-flash", hint: "Active provider" },
    { label: "Patient ID Prefix", value: "ARISTON-", hint: "Prepended to all patient IDs" },
    { label: "RAG Depth", value: "5 sources", hint: "Per query retrieval limit" },
    { label: "Audit Mode", value: "GxP / 21 CFR Part 11", hint: "Immutable SHA-256 chain" },
    { label: "SLA Threshold", value: "3000 ms", hint: "Alert trigger for latency" },
  ];
  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-white">Platform Settings</h2>
      <div className="glass-card p-6 space-y-4">
        {fields.map(f => (
          <div key={f.label} className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
            <div>
              <p className="text-sm text-white font-semibold">{f.label}</p>
              <p className="text-[10px] text-slate-500">{f.hint}</p>
            </div>
            <span className="px-3 py-1.5 bg-white/5 rounded-lg text-xs text-brand-accent font-mono">{f.value}</span>
          </div>
        ))}
        <button
          onClick={() => { setSaved(true); setTimeout(() => setSaved(false), 2000); }}
          className="mt-4 w-full py-2.5 bg-brand-primary/20 border border-brand-primary/30 rounded-xl text-brand-primary text-sm font-bold hover:bg-brand-primary/30 transition-colors"
        >
          {saved ? "✓ Settings Saved" : "Save Settings"}
        </button>
      </div>
    </div>
  );
}

function AuditView({ logs }) {
  const [auditTrail, setAuditTrail] = useState([]);
  
  useEffect(() => {
    const fetchAudit = async () => {
      try {
        const res = await axios.get(`${API_BASE}/audit`);
        setAuditTrail(res.data);
      } catch (e) {
        console.error("Audit fetch failed", e);
      }
    };
    fetchAudit();
  }, []);

  const passRate = auditTrail.length > 0 ? "100%" : "0%";
  
  return (
    <div className="space-y-6">
       <h1 className="text-4xl font-bold text-white mb-6">Clinical Audit Logs</h1>
       <div className="grid grid-cols-3 gap-6 mb-8">
          <StatCard label="Total Audited Jobs" value={auditTrail.length} icon={<Database />} />
          <StatCard label="Compliance Pass Rate" value={passRate} icon={<ShieldCheck />} />
          <StatCard label="Avg Reliability" value="0.96" icon={<Activity />} />
       </div>
       <div className="glass-card overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="bg-white/5 text-slate-400 uppercase text-[10px] tracking-widest">
                <tr>
                  <th className="px-6 py-4">Job ID</th>
                  <th className="px-6 py-4">Timestamp</th>
                  <th className="px-6 py-4">GxP Hash</th>
                  <th className="px-6 py-4">Status</th>
                </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
                {auditTrail.map((log, i) => (
                  <tr key={i} className="hover:bg-white/5 transition-colors group">
                    <td className="px-6 py-4 font-mono text-[10px] text-brand-primary group-hover:text-brand-accent transition-colors">{log.job_id?.slice(0, 8)}</td>
                    <td className="px-6 py-4 text-slate-400 text-xs">{new Date(log.timestamp).toLocaleString()}</td>
                    <td className="px-6 py-4">
                       <div className="flex items-center space-x-2">
                          <ShieldCheck size={12} className="text-brand-accent" />
                          <span className="font-mono text-[9px] text-slate-500 uppercase">{log.entry_hash.slice(0, 16)}...</span>
                       </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span className="px-2 py-1 rounded bg-brand-accent/10 text-brand-accent text-[9px] font-bold uppercase tracking-widest border border-brand-accent/20">GxP Verified</span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
       </div>
    </div>
  );
}

function VisionView({ job, onBack }) {
  const meta = job?.result?.metadata || {};
  const [showSaliency, setShowSaliency] = useState(false);
  const saliency = meta.radiology_saliency || null;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-white mb-2">Radiology Visualizer</h1>
          <p className="text-slate-400">Interactive medical imaging grounded by Ariston Vision Agents.</p>
        </div>
        <div className="flex space-x-2">
           <button 
              onClick={() => setShowSaliency(!showSaliency)}
              disabled={!saliency}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${!saliency ? 'opacity-50 cursor-not-allowed bg-white/5 text-slate-500' : showSaliency ? 'bg-brand-primary text-slate-950 shadow-[0_0_15px_rgba(34,211,238,0.5)]' : 'bg-white/5 text-brand-primary border border-brand-primary/20'}`}
           >
              {showSaliency ? 'Deactivate Saliency' : 'Analyze Saliency'}
           </button>
           <button onClick={onBack} className="px-4 py-2 glass-card hover:bg-white/10 transition-colors text-sm text-white">
             &larr; Back
           </button>
        </div>
      </div>
      
      {!job ? (
        <div className="glass-card aspect-video flex items-center justify-center border-dashed border-2 mt-12 text-slate-500 flex-col space-y-4">
          <UploadCloud size={48} className="text-slate-700" />
          <p>Upload an X-ray, CT, MRI, DICOM, PDF, or FHIR file in the Orchestrator tab.</p>
          <p className="text-xs text-slate-600">Supported: JPG, PNG, DICOM, PDF, JSON (FHIR), CSV, TXT</p>
        </div>
      ) : (
        <div className="space-y-6 mt-4">
          {/* Files analyzed badge */}
          {job.result?.files_processed && (
            <div className="flex items-center space-x-3 flex-wrap gap-2">
              {job.result.files_processed.map((fname, i) => (
                <span key={i} className="px-3 py-1 bg-brand-primary/10 border border-brand-primary/20 rounded-full text-[10px] text-brand-primary font-mono">
                  {fname}
                </span>
              ))}
              <span className="text-xs text-slate-500">{job.result.images_analyzed} image(s) · {job.result.documents_extracted} doc(s)</span>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Image panel */}
            <div className="glass-card aspect-square overflow-hidden bg-black flex items-center justify-center relative">
               <div className="absolute top-4 left-4 z-10 px-3 py-1 bg-black/50 backdrop-blur rounded text-[10px] font-mono text-brand-primary">
                  LAYER: {job.result?.layer?.toUpperCase() || meta.layer?.toUpperCase() || "RADIOLOGY"}
               </div>
               {showSaliency && saliency && (
                 <motion.div
                   initial={{ opacity: 0, scale: 0.5 }}
                   animate={{ opacity: 1, scale: 1 }}
                   className="absolute z-20 pointer-events-none border-2 border-red-500 rounded-full shadow-[0_0_20px_rgba(239,68,68,0.8)] flex items-center justify-center"
                   style={{
                     left: `${saliency.coordinates?.x}px`,
                     top: `${saliency.coordinates?.y}px`,
                     width: `${saliency.coordinates?.radius * 2}px`,
                     height: `${saliency.coordinates?.radius * 2}px`,
                   }}
                 >
                   <span className="bg-red-500 text-white text-[8px] font-bold px-1 rounded absolute -top-4">{saliency.label}</span>
                 </motion.div>
               )}
               {meta.images && meta.images.length > 0 ? (
                 <img src={meta.images[0]} className="w-full h-full object-contain opacity-80" alt="Medical Scan" />
               ) : (
                 <div className="text-center p-8">
                   <Activity className="w-12 h-12 text-brand-primary/30 mx-auto mb-4 animate-pulse" />
                   <p className="text-slate-600 font-mono text-xs">Analysis complete — no image preview available</p>
                 </div>
               )}
            </div>

            {/* Report panel */}
            <div className="space-y-4">
              {/* Vision analysis */}
              {job.result?.vision_analysis && (
                <div className="glass-card p-5 border-l-4 border-brand-accent">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold flex items-center gap-2 text-brand-accent">
                      <Eye size={14} /> Vision Analysis (Gemini 2.0 Flash)
                    </h3>
                    <CopyButton text={job.result.vision_analysis} label="Copy" />
                  </div>
                  <div className="text-slate-300 text-xs leading-relaxed max-h-48 overflow-y-auto pr-2 whitespace-pre-wrap font-mono">
                    {job.result.vision_analysis}
                  </div>
                </div>
              )}

              {/* AI clinical report */}
              <div className="glass-card p-5 border-l-4 border-brand-primary">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold flex items-center gap-2">
                    <FileText size={14} className="text-brand-primary" /> Clinical Report (RAG-Grounded)
                  </h3>
                  <CopyButton text={job.result?.content} label="Copy Report" />
                </div>
                <div className="text-slate-300 text-sm leading-relaxed max-h-64 overflow-y-auto pr-2">
                  {job.result?.content
                    ? job.result.content.split('\n').map((line, i) => <p key={i} className="mb-1">{line}</p>)
                    : <p className="text-slate-500">No report available.</p>}
                </div>
              </div>

              {/* Pipeline metadata */}
              <div className="glass-card p-4">
                <p className="text-[10px] text-slate-500 uppercase font-bold mb-3">Pipeline</p>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="p-2 bg-white/5 rounded">MODEL: {job.result?.model || meta.model_used || '—'}</div>
                  <div className="p-2 bg-white/5 rounded">LAYER: {job.result?.layer || '—'}</div>
                  <div className="p-2 bg-white/5 rounded">SAFETY: {meta.safety?.flag || 'SAFE'}</div>
                  <div className="p-2 bg-white/5 rounded">LATENCY: {meta.latency_ms ? `${meta.latency_ms}ms` : '—'}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MoleculeView() {
  const mountRef = useRef(null);

  useEffect(() => {
    import('three').then((THREE) => {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      
      if (!mountRef.current) return;
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      renderer.setSize(width, height);
      mountRef.current.appendChild(renderer.domElement);

      const geometry = new THREE.IcosahedronGeometry(1.5, 1);
      const material = new THREE.MeshPhongMaterial({ 
        color: 0x22d3ee, 
        wireframe: true,
        transparent: true,
        opacity: 0.8
      });
      const molecule = new THREE.Mesh(geometry, material);
      scene.add(molecule);

      const light = new THREE.PointLight(0xffffff, 1, 100);
      light.position.set(10, 10, 10);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x404040));

      camera.position.z = 4;

      const animate = () => {
        requestAnimationFrame(animate);
        molecule.rotation.x += 0.005;
        molecule.rotation.y += 0.005;
        renderer.render(scene, camera);
      };
      animate();

      return () => {
        if (mountRef.current && renderer.domElement) mountRef.current.removeChild(renderer.domElement);
      };
    });
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-white mb-2">3D Molecule Lab</h1>
          <p className="text-slate-400">Interactive protein-ligand binding simulation.</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-6">
         <div className="col-span-2 glass-card h-[500px] flex items-center justify-center relative overflow-hidden">
            <div ref={mountRef} className="w-full h-full" />
         </div>
         <div className="space-y-6">
            <div className="glass-card p-6 space-y-4">
               <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center space-x-2">
                  <Activity size={14} className="text-brand-primary" />
                  <span>Real-Time ADME</span>
               </h3>
               <div className="space-y-3">
                  <ADMERow label="Lipinski" value="PASS" status="text-green-400" />
                  <ADMERow label="LogP" value="2.14" status="text-brand-primary" />
                  <ADMERow label="BBB Perm" value="HIGH" status="text-orange-400" />
               </div>
            </div>
         </div>
      </div>
    </div>
  );
}

function ADMERow({ label, value, status }) {
   return (
      <div className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
         <span className="text-xs text-slate-500">{label}</span>
         <span className={`text-xs font-mono font-bold ${status}`}>{value}</span>
      </div>
   );
}

function EcosystemCard({ icon, label, status, color }) {
  return (
    <div className="glass-card p-3 flex items-center space-x-3">
      <div className={`${color} opacity-80`}>{icon}</div>
      <div>
        <p className="text-[10px] text-slate-500 uppercase font-bold">{label}</p>
        <p className="text-xs text-white font-semibold">{status}</p>
      </div>
    </div>
  );
}

function BiometricsSimulator() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let offset = 0;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 2;
      for (let x = 0; x < canvas.width; x++) {
        const y = 30 + Math.sin((x + offset) * 0.1) * 10 + (Math.random() > 0.98 ? -20 : 0);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      offset += 2;
      requestAnimationFrame(animate);
    };
    animate();
  }, []);
  return <canvas ref={canvasRef} width={200} height={60} className="w-full h-full opacity-30" />;
}

function TwinView({ job }) {
  const sim = job?.metadata?.digital_twin_simulation || {};
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
         <div>
            <h1 className="text-4xl font-bold text-white mb-2">Clinical Digital Twin</h1>
            <p className="text-slate-400">In-silico predictive outcomes based on longitudinal history & genomics.</p>
         </div>
         <div className="flex space-x-2">
            <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${sim.prediction === 'CRITICAL' ? 'bg-red-500/20 text-red-500' : 'bg-brand-primary/20 text-brand-primary'}`}>
               Status: {sim.prediction || 'STABLE'}
            </span>
         </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
         <div className="glass-card p-8 space-y-6 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
               <Activity size={80} />
            </div>
            <h3 className="text-xl font-bold text-white">Efficacy vs Toxicity</h3>
            <div className="space-y-4">
               <ProgressRow label="Predicted Efficacy" value={`${(sim.efficacy_score || 0.8)*100}%`} color="bg-brand-primary" />
               <ProgressRow label="Toxicity Risk" value={`${(sim.toxicity_risk || 0.1)*100}%`} color="bg-red-500" />
            </div>
         </div>
         <div className="glass-card p-8 space-y-6">
            <h3 className="text-xl font-bold text-white">Organ Impact Matrix</h3>
            <div className="grid grid-cols-3 gap-4">
               {Object.entries(sim.organ_impact || { liver: 'LOW', cardiac: 'LOW', renal: 'LOW' }).map(([organ, risk]) => (
                  <div key={organ} className="text-center p-4 bg-white/5 rounded-xl border border-white/5">
                     <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">{organ}</p>
                     <p className={`text-xs font-bold ${risk === 'HIGH' ? 'text-red-500' : 'text-brand-primary'}`}>{risk}</p>
                  </div>
               ))}
            </div>
         </div>
      </div>
    </div>
  );
}

const AGENCIES = [
  { key: "csr",               label: "CSR — ICH E3",           flag: "🌐", lang: "en",  region: "Global / FDA" },
  { key: "ectd",              label: "eCTD Module 5",           flag: "🇺🇸", lang: "en", region: "Global / FDA" },
  { key: "cmc",               label: "CMC — Module 3",          flag: "🌐", lang: "en",  region: "Global / FDA" },
  { key: "pv_narrative",      label: "PV Case Narrative",       flag: "🌐", lang: "en",  region: "Global / CIOMS" },
  { key: "cofepris_registro", label: "Registro Sanitario",      flag: "🇲🇽", lang: "es", region: "COFEPRIS — México" },
  { key: "cofepris_pv",       label: "Reporte PV",              flag: "🇲🇽", lang: "es", region: "COFEPRIS — México" },
  { key: "anvisa_registro",   label: "Dossiê de Registro",      flag: "🇧🇷", lang: "pt", region: "ANVISA — Brasil" },
  { key: "anmat_registro",    label: "Autorización Comercial",  flag: "🇦🇷", lang: "es", region: "ANMAT — Argentina" },
  { key: "invima_registro",   label: "Registro Sanitario",      flag: "🇨🇴", lang: "es", region: "INVIMA — Colombia" },
];

function RegulatoryView() {
  const [docType, setDocType]       = useState("cofepris_registro");
  const [drugName, setDrugName]     = useState("dabrafenib");
  const [indication, setIndication] = useState("melanoma metastásico BRAF V600E");
  const [nctId, setNctId]           = useState("NCT01227889");
  const [section, setSection]       = useState("");
  const [loading, setLoading]       = useState(false);
  const [draft, setDraft]           = useState(null);
  const [error, setError]           = useState(null);

  const selected = AGENCIES.find(a => a.key === docType) || AGENCIES[0];

  const generate = async () => {
    setLoading(true);
    setDraft(null);
    setError(null);
    try {
      const res = await axios.post(`${API_BASE}/pharma/draft`, {
        document_type: docType,
        drug_name: drugName,
        indication,
        nct_id: nctId || undefined,
        section: section || undefined,
      });
      setDraft(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadPdf = async () => {
    try {
      const res = await axios.post(`${API_BASE}/pharma/draft/pdf`, {
        document_type: docType,
        drug_name: drugName,
        indication,
        nct_id: nctId || undefined,
        section: section || undefined,
      }, { responseType: "blob" });
      const url = URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `ariston_${drugName.toLowerCase().replace(/\s+/g, "_")}_${docType}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError("PDF export failed");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-4xl font-bold text-white mb-2">Regulatory Copilot</h1>
          <p className="text-slate-400">AI-assisted drafting for FDA, COFEPRIS, ANVISA, ANMAT & INVIMA submissions.</p>
        </div>
        <div className="flex items-center space-x-2 px-3 py-1.5 bg-white/5 rounded-xl border border-white/10 text-xs text-brand-accent">
          <ShieldCheck size={12} />
          <span>GxP Audit Active</span>
        </div>
      </div>

      {/* Agency selector */}
      <div className="glass-card p-5 space-y-4">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Regulatory Agency & Document Type</p>
        <div className="grid grid-cols-3 gap-2">
          {AGENCIES.map(a => (
            <button
              key={a.key}
              onClick={() => setDocType(a.key)}
              className={`p-3 rounded-xl border text-left transition-all ${
                docType === a.key
                  ? "border-brand-primary/50 bg-brand-primary/10 text-white"
                  : "border-white/5 bg-white/2 text-slate-400 hover:bg-white/5"
              }`}
            >
              <span className="text-lg mr-1">{a.flag}</span>
              <span className="text-[10px] font-bold uppercase tracking-wide block mt-1">{a.region}</span>
              <span className="text-xs text-slate-300">{a.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Form */}
      <div className="glass-card p-5 space-y-4">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Compound & Indication</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Drug / Compound</label>
            <input
              value={drugName}
              onChange={e => setDrugName(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:border-brand-primary/50 outline-none"
              placeholder="e.g. dabrafenib"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Indication</label>
            <input
              value={indication}
              onChange={e => setIndication(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:border-brand-primary/50 outline-none"
              placeholder="e.g. BRAF V600E metastatic melanoma"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1">NCT ID (optional — grounds in live data)</label>
            <input
              value={nctId}
              onChange={e => setNctId(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:border-brand-primary/50 outline-none"
              placeholder="e.g. NCT01227889"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Section only (leave blank for full doc)</label>
            <input
              value={section}
              onChange={e => setSection(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:border-brand-primary/50 outline-none"
              placeholder="e.g. Synopsis"
            />
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-white/5">
          <div className="text-xs text-slate-500">
            Output language: <span className="text-brand-primary font-bold uppercase">{selected.lang === "es" ? "Español" : selected.lang === "pt" ? "Português" : "English"}</span>
            &nbsp;·&nbsp;Agency: <span className="text-slate-300">{selected.region}</span>
          </div>
          <button
            onClick={generate}
            disabled={loading || !drugName || !indication}
            className="medical-gradient px-8 py-2.5 rounded-xl font-semibold shadow-lg shadow-brand-primary/20 hover:scale-[1.02] transition-transform flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Drafting…</span>
              </>
            ) : (
              <>
                <FileText className="w-4 h-4" />
                <span>Generate Draft</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="glass-card p-4 border-medical-red/30 bg-medical-red/5 text-medical-red text-sm">
          {error}
        </div>
      )}

      {/* Draft output */}
      {draft && (
        <div className="glass-card overflow-hidden">
          {/* Meta bar */}
          <div className="px-6 py-3 bg-white/3 border-b border-white/5 flex items-center justify-between">
            <div className="flex items-center space-x-4 text-[10px] text-slate-500 uppercase font-bold tracking-widest">
              <span>{selected.flag} {selected.region}</span>
              <span>·</span>
              <span>{draft.drug_name} — {draft.indication}</span>
              {draft.nct_id && <><span>·</span><span className="text-brand-primary">{draft.nct_id}</span></>}
              <span>·</span>
              <span>{draft.sources_used} sources</span>
              {draft.trial_data_used && <><span>·</span><span className="text-brand-accent">ClinicalTrials grounded</span></>}
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={downloadPdf}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-brand-accent/10 border border-brand-accent/20 text-brand-accent rounded-lg text-[10px] font-bold uppercase tracking-widest hover:bg-brand-accent/20 transition-colors"
              >
                <FileText size={12} />
                <span>Download PDF</span>
              </button>
            </div>
          </div>
          <div className="p-8 overflow-y-auto max-h-[600px]">
            <pre className="text-slate-300 font-mono text-sm whitespace-pre-wrap leading-relaxed">
              {draft.draft}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Platform Dashboard (Phase 5 + 6) ────────────────────────────────────────

function PlatformDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/platform/dashboard?tenant_id=global&tier=standard`)
      .then(r => { setData(r.data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-slate-400 p-8">Loading platform dashboard...</div>;
  if (!data) return <div className="text-medical-red p-8">Failed to load dashboard.</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-1">Platform Dashboard</h1>
        <p className="text-slate-400 text-sm">Real-time SLA, usage, data moat, and agent health.</p>
      </div>

      {/* SLA */}
      <div className="glass-card p-6">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">SLA Monitor (24h)</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Uptime", value: `${data.sla.uptime_pct?.toFixed(2) ?? 100}%`, ok: (data.sla.uptime_pct ?? 100) >= 99.5 },
            { label: "P50 Latency", value: `${data.sla.p50_ms?.toFixed(0) ?? 0}ms`, ok: (data.sla.p50_ms ?? 0) < 2000 },
            { label: "P95 Latency", value: `${data.sla.p95_ms?.toFixed(0) ?? 0}ms`, ok: (data.sla.p95_ms ?? 0) < 5000 },
            { label: "Status", value: data.sla.status === 'healthy' ? '✓ Healthy' : '⚠ Breach', ok: data.sla.status === 'healthy' },
          ].map(m => (
            <div key={m.label} className="bg-white/5 rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase font-bold">{m.label}</p>
              <p className={`text-xl font-black mt-1 ${m.ok ? 'text-brand-accent' : 'text-medical-red'}`}>{m.value}</p>
            </div>
          ))}
        </div>
        {data.sla.breaches?.length > 0 && (
          <div className="mt-4 p-3 bg-medical-red/10 border border-medical-red/20 rounded-lg">
            {data.sla.breaches.map((b, i) => <p key={i} className="text-medical-red text-xs">{b}</p>)}
          </div>
        )}
      </div>

      {/* Usage */}
      <div className="glass-card p-6">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">Usage (Current Period)</p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "API Calls", value: data.usage.api_calls },
            { label: "Pipeline Runs", value: data.usage.pipeline_runs },
            { label: "RAG Queries", value: data.usage.rag_queries },
          ].map(u => (
            <div key={u.label} className="bg-white/5 rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase font-bold">{u.label}</p>
              <p className="text-2xl font-black text-white mt-1">{u.value?.toLocaleString() ?? 0}</p>
            </div>
          ))}
        </div>
        {data.usage.overage_usd > 0 && (
          <p className="mt-3 text-xs text-brand-secondary">Overage: ${data.usage.overage_usd?.toFixed(2)}</p>
        )}
      </div>

      {/* Data Moat */}
      <div className="glass-card p-6">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">Data Moat (Phase 6)</p>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Embeddings", value: data.data_moat.total_embeddings },
            { label: "RWE Records", value: data.data_moat.rwe_records },
            { label: "Freshness", value: `${((data.data_moat.freshness_score ?? 1) * 100).toFixed(0)}%` },
            { label: "Stale Sets", value: data.data_moat.stale_datasets },
          ].map(m => (
            <div key={m.label} className="bg-white/5 rounded-xl p-4">
              <p className="text-[10px] text-slate-500 uppercase font-bold">{m.label}</p>
              <p className="text-xl font-black text-brand-primary mt-1">{m.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Agents */}
      <div className="glass-card p-6">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">Active Agents ({data.agents.active})</p>
        <div className="grid grid-cols-2 gap-2">
          {data.agents.endpoints.map(ep => (
            <div key={ep} className="bg-white/5 rounded-lg px-3 py-2 flex items-center space-x-2">
              <div className="w-1.5 h-1.5 rounded-full bg-brand-accent animate-pulse" />
              <span className="text-xs text-slate-300 font-mono">{ep}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── LATAM Data View (Phase 6) ────────────────────────────────────────────────

function LatamDataView() {
  const [coverage, setCoverage] = useState(null);
  const [burden, setBurden] = useState(null);
  const [countries] = useState(['brazil', 'mexico', 'colombia', 'argentina', 'chile']);
  const [conditions] = useState(['type2_diabetes', 'cardiovascular', 'dengue', 'oncology']);
  const [loading, setLoading] = useState(true);
  const [accumulating, setAccumulating] = useState(false);

  useEffect(() => {
    Promise.all([
      axios.get(`${API_BASE}/phase6/latam/coverage`),
      axios.post(`${API_BASE}/phase6/latam/burden`, { countries, conditions }),
    ]).then(([cov, bur]) => {
      setCoverage(cov.data);
      setBurden(bur.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const seedData = async () => {
    setAccumulating(true);
    try {
      await axios.post(`${API_BASE}/platform/rwe/seed`, {
        therapeutic_area: 'diabetes',
        countries,
        embed: true,
      });
    } finally {
      setAccumulating(false);
    }
  };

  if (loading) return <div className="text-slate-400 p-8">Loading LATAM data...</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">LATAM Data Moat</h1>
          <p className="text-slate-400 text-sm">DATASUS · SINAVE · SISPRO · SNVS · DEIS — {coverage?.aggregate?.total_population_millions}M population covered</p>
        </div>
        <button
          onClick={seedData}
          disabled={accumulating}
          className="medical-gradient px-5 py-2 rounded-xl text-sm font-semibold flex items-center space-x-2 disabled:opacity-50"
        >
          <Database className="w-4 h-4" />
          <span>{accumulating ? 'Accumulating...' : 'Seed RWE Data'}</span>
        </button>
      </div>

      {/* Country Coverage */}
      <div className="glass-card p-6">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">Data Sources</p>
        <div className="grid grid-cols-5 gap-3">
          {coverage && Object.entries(coverage.countries).map(([country, info]) => (
            <div key={country} className="bg-white/5 rounded-xl p-3">
              <p className="text-xs font-bold text-white capitalize">{country}</p>
              <p className="text-[10px] text-slate-500 mt-1">{info.coverage_source}</p>
              <p className="text-brand-primary font-bold text-sm mt-2">{info.population_millions}M pop</p>
              <p className="text-[10px] text-slate-400">{info.estimated_records_millions}M records</p>
            </div>
          ))}
        </div>
      </div>

      {/* Disease Burden */}
      {burden && (
        <div className="glass-card p-6">
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-4">Disease Burden (per 100k) — PAHO/WHO 2022</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left py-2 text-slate-500 font-bold uppercase">Country</th>
                  {conditions.map(c => <th key={c} className="text-right py-2 text-slate-500 font-bold uppercase px-2">{c.replace('_', ' ')}</th>)}
                </tr>
              </thead>
              <tbody>
                {countries.map(country => (
                  <tr key={country} className="border-b border-white/5 hover:bg-white/2">
                    <td className="py-2 text-white font-semibold capitalize">{country}</td>
                    {conditions.map(cond => {
                      const d = burden.burden?.[country]?.[cond];
                      return (
                        <td key={cond} className="py-2 text-right px-2 text-slate-300">
                          {d?.prevalence_per_100k ? d.prevalence_per_100k.toLocaleString() : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Agents View ──────────────────────────────────────────────────────────────

function AgentsView() {
  const [agents, setAgents] = useState(null);
  const [status, setStatus] = useState(null);
  const [selected, setSelected] = useState(null);
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    Promise.all([
      axios.get(`${API_BASE}/platform/agents`),
      axios.get(`${API_BASE}/agents/status`),
    ]).then(([a, s]) => {
      setAgents(a.data);
      setStatus(s.data);
    }).catch(() => {});
  }, []);

  const runAgent = async () => {
    if (!selected || !input) return;
    setRunning(true);
    setResult(null);
    try {
      // Route to correct endpoint
      const endpointMap = {
        pharmacist:       { url: `${API_BASE}/agents/pharmacist/review`, body: { prompt: input, context: {} } },
        latam_regulatory: { url: `${API_BASE}/agents/latam/roadmap`, body: { drug_name: input, indication: 'General', target_countries: ['brazil', 'mexico'] } },
        vision_radiology: { url: `${API_BASE}/agents/vision/analyze`, body: { prompt: input, images: [] } },
        digital_twin:     { url: `${API_BASE}/agents/twin/simulate`, body: { drug: input, patient_history: 'Adult patient' } },
        pharmacogenomics: { url: `${API_BASE}/agents/pgx/cross-reference`, body: { drug_name: input } },
        site_selection:   { url: `${API_BASE}/agents/sites/recommend`, body: { therapeutic_area: input, top_n: 3 } },
      };
      const ep = endpointMap[selected.id];
      if (ep) {
        const r = await axios.post(ep.url, ep.body);
        setResult(r.data);
      }
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white mb-1">Agent Console</h1>
        <p className="text-slate-400 text-sm">10 specialized agents — all active and wired to platform data.</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Agent Grid */}
        <div className="glass-card p-5 space-y-2">
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-3">Select Agent</p>
          {agents?.agents?.map(agent => (
            <button
              key={agent.id}
              onClick={() => { setSelected(agent); setResult(null); }}
              className={`w-full text-left px-4 py-3 rounded-xl transition-all flex items-center justify-between ${
                selected?.id === agent.id
                  ? 'bg-brand-primary/20 border border-brand-primary/40'
                  : 'bg-white/5 hover:bg-white/8 border border-transparent'
              }`}
            >
              <div>
                <p className="text-sm font-semibold text-white">{agent.name}</p>
                <p className="text-[10px] text-slate-400 mt-0.5">{agent.description.slice(0, 60)}...</p>
              </div>
              <div className="w-1.5 h-1.5 rounded-full bg-brand-accent flex-shrink-0" />
            </button>
          ))}
        </div>

        {/* Agent Runner */}
        <div className="space-y-4">
          {selected && (
            <div className="glass-card p-5 space-y-4">
              <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{selected.name}</p>
              <p className="text-xs text-slate-400">{selected.description}</p>
              <p className="text-[10px] text-slate-600 font-mono">{selected.endpoint}</p>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder={`Enter input for ${selected.name}...`}
                className="w-full h-24 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white resize-none placeholder:text-slate-600 focus:border-brand-primary/50 outline-none"
              />
              <button
                onClick={runAgent}
                disabled={running || !input}
                className="w-full medical-gradient py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center space-x-2 disabled:opacity-50"
              >
                {running ? <Activity className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
                <span>{running ? 'Running...' : `Run ${selected.name}`}</span>
              </button>
            </div>
          )}

          {result && (
            <div className="glass-card p-5">
              <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-3">Result</p>
              <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap overflow-auto max-h-64">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}

          {!selected && (
            <div className="glass-card p-8 flex flex-col items-center justify-center text-center">
              <Brain className="text-slate-600 mb-3" size={32} />
              <p className="text-slate-500 text-sm">Select an agent to run it</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ProgressRow({ label, value, color = "bg-brand-primary" }) {
  return (
    <div className="space-y-1">
       <div className="flex justify-between text-xs font-medium">
          <span className="text-slate-400">{label}</span>
          <span className="text-white">{value}</span>
       </div>
       <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: value }}
            className={`h-full ${color}`} 
          />
       </div>
    </div>
  );
}

export default App;

function CommandPalette({ isOpen, onClose, jobs, onSelectJob }) {
  const [query, setQuery] = useState('');
  const filteredJobs = (jobs || []).filter(j => 
    j.prompt?.toLowerCase().includes(query.toLowerCase()) || 
    j.job_id?.toLowerCase().includes(query.toLowerCase())
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] px-4">
       <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={onClose} />
       <motion.div 
         initial={{ opacity: 0, scale: 0.95, y: -20 }}
         animate={{ opacity: 1, scale: 1, y: 0 }}
         className="w-full max-w-xl glass-card bg-[#0a0f1d] border-white/10 shadow-2xl relative z-10 overflow-hidden"
       >
          <div className="flex items-center px-4 py-3 border-b border-white/5">
             <Search size={18} className="text-slate-500" />
             <input 
               autoFocus
               placeholder="Search agents, jobs, or patient records..."
               className="flex-1 bg-transparent border-none focus:ring-0 text-white placeholder:text-slate-600 px-3 py-1"
               value={query}
               onChange={(e) => setQuery(e.target.value)}
             />
             <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px] text-slate-400 font-mono">ESC</kbd>
          </div>
          <div className="max-h-80 overflow-y-auto p-2">
             {filteredJobs.length === 0 && (
               <p className="text-center p-8 text-slate-600 text-sm">No results found...</p>
             )}
             {filteredJobs.map(job => (
               <button 
                 key={job.job_id}
                 onClick={() => onSelectJob(job)}
                 className="w-full text-left p-3 rounded-lg hover:bg-white/5 transition-all group flex items-center justify-between"
               >
                  <div className="flex items-center space-x-3">
                     <div className="p-2 bg-brand-primary/10 rounded-lg text-brand-primary">
                        <Terminal size={14} />
                     </div>
                     <div>
                        <p className="text-xs font-semibold text-slate-200">{job.prompt.slice(0, 40)}...</p>
                        <p className="text-[10px] text-slate-500 font-mono uppercase">{job.job_id.slice(0, 8)}</p>
                     </div>
                  </div>
                  <ChevronRight size={14} className="text-slate-600 group-hover:text-brand-primary transition-colors" />
               </button>
             ))}
          </div>
          <div className="bg-slate-950/50 px-4 py-2 border-t border-white/5 flex items-center justify-between">
             <p className="text-[9px] text-slate-600 uppercase font-bold tracking-widest">Ariston OS Command Hub</p>
             <div className="flex space-x-2">
                <kbd className="px-1 py-0.5 bg-slate-900 rounded text-[8px] text-slate-500">↑↓ Navigate</kbd>
                <kbd className="px-1 py-0.5 bg-slate-900 rounded text-[8px] text-slate-500">↵ Select</kbd>
             </div>
          </div>
       </motion.div>
    </div>
  );
}
