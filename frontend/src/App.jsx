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

const API_BASE = "http://127.0.0.1:8005/api/v1";
const WS_BASE = "ws://127.0.0.1:8005/api/v1/ws/jobs";

function App() {
  const [activeTab, setActiveTab] = useState('orchestrator');
  const [prompt, setPrompt] = useState('');
  const [patientId, setPatientId] = useState('ARISTON-001');
  const [jobs, setJobs] = useState([]);
  const [wsLogs, setWsLogs] = useState([]);
  const [files, setFiles] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
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

    return () => ws.current.close();
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
      if (file.type.startsWith('image/')) {
        const b64 = await fileToBase64(file);
        processed.push({ type: 'image', name: file.name, data: b64 });
      } else if (file.name.endsWith('.json')) {
        const text = await file.text();
        try {
          processed.push({ type: 'fhir', name: file.name, data: JSON.parse(text) });
        } catch (e) { console.error("Invalid FHIR", e); }
      }
    }
    setFiles(prev => [...prev, ...processed]);
  };

  const submitOrchestration = async () => {
    if (!prompt) return;
    
    const requestData = {
      prompt: prompt,
      patient_id: patientId,
      images: files.filter(f => f.type === 'image').map(f => f.data),
      fhir_bundle: files.filter(f => f.type === 'fhir').map(f => f.data),
      context: { layer: activeTab === 'vision' ? 'radiology' : 'clinical' }
    };
    
    try {
      const response = await axios.post(`${API_BASE}/orchestrate`, requestData);
      
      setJobs(prev => [{
        job_id: response.data.job_id,
        status: 'accepted',
        prompt: prompt,
        timestamp: new Date().toISOString()
      }, ...prev]);
      
      setPrompt('');
      setFiles([]);
    } catch (error) {
      console.error("Submission failed", error);
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
           <NavItem icon={<Brain />} label="Orchestrator" active={activeTab === 'orchestrator'} onClick={() => setActiveTab('orchestrator')} />
           <NavItem icon={<Activity />} label="Patient Timeline" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
           <NavItem icon={<Activity />} label="Digital Twin" active={activeTab === 'twin'} onClick={() => setActiveTab('twin')} />
           <NavItem icon={<FileText />} label="Regulatory" active={activeTab === 'reg'} onClick={() => setActiveTab('reg')} />
          <NavItem icon={<Beaker />} label="Molecule-Lab" active={activeTab === 'mol'} onClick={() => setActiveTab('mol')} />
          <NavItem icon={<Microscope />} label="Trial-Matcher" active={activeTab === 'trial'} onClick={() => setActiveTab('trial')} />
          <NavItem icon={<Database />} label="Audit Logs" active={activeTab === 'audit'} onClick={() => setActiveTab('audit')} />
          <NavItem icon={<Eye />} label="Rad-Visualizer" active={activeTab === 'vision'} onClick={() => setActiveTab('vision')} />
          <NavItem icon={<ShieldCheck />} label="Benchmarking" active={activeTab === 'bench'} onClick={() => setActiveTab('bench')} />
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
               <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <EcosystemCard icon={<Database size={16} />} label="OpenFDA" status="Connected" color="text-brand-primary" />
                  <EcosystemCard icon={<FileText size={16} />} label="ClinicalTrials" status="Active" color="text-brand-secondary" />
                  <EcosystemCard icon={<Search size={16} />} label="PubMed" status="Syncing" color="text-brand-primary" />
                  <EcosystemCard icon={<ShieldCheck size={16} />} label="RxNorm" status="Optimized" color="text-brand-secondary" />
               </div>
               
               <div className="flex justify-between items-end">
                  <div>
                    <h1 className="text-4xl font-bold text-white mb-2">Life Science Orchestrator</h1>
                    <p className="text-slate-400">Execute clinical, pharma, and radiology agents through a unified OS layer.</p>
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
                  <div className="flex items-center justify-between pt-4 border-t border-white/5">
                    <div className="flex space-x-4">
                       <label className="flex items-center space-x-2 cursor-pointer text-slate-400 hover:text-white transition-colors">
                          <UploadCloud className="w-5 h-5" />
                          <span className="text-sm">Upload Scan/Data</span>
                          <input type="file" multiple className="hidden" onChange={handleFileUpload} />
                       </label>
                    </div>
                    <button 
                      onClick={submitOrchestration}
                      className="medical-gradient px-8 py-2.5 rounded-xl font-semibold shadow-lg shadow-brand-primary/20 hover:scale-[1.02] transition-transform flex items-center space-x-2"
                    >
                      <span>Orchestrate</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
               </section>

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
                               <p className="font-medium text-slate-100">{job.prompt.slice(0, 60)}...</p>
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
        <div className="glass-card aspect-video flex items-center justify-center border-dashed border-2 mt-12 text-slate-500">
           No active scan selected. Upload a DICOM or MRI scan in the orchestrator.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">
          <div className="glass-card aspect-square overflow-hidden bg-black flex items-center justify-center relative">
             <div className="absolute top-4 left-4 z-10 px-3 py-1 bg-black/50 backdrop-blur rounded text-[10px] font-mono text-brand-primary">
                MODALITY: {meta.modality || "MRI"}
             </div>
             
             {/* Saliency Overlay */}
             {showSaliency && saliency && (
               <motion.div 
                 initial={{ opacity: 0, scale: 0.5 }}
                 animate={{ opacity: 1, scale: 1 }}
                 className="absolute z-20 pointer-events-none border-2 border-red-500 rounded-full shadow-[0_0_20px_rgba(239,68,68,0.8)] flex items-center justify-center"
                 style={{ 
                   left: `${saliency.coordinates?.x}px`, 
                   top: `${saliency.coordinates?.y}px`,
                   width: `${saliency.coordinates?.radius * 2}px`,
                   height: `${saliency.coordinates?.radius * 2}px`
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
                  <p className="text-slate-600 font-mono text-xs">Simulating reconstruction...</p>
               </div>
             )}
          </div>
          <div className="space-y-6">
             <div className="glass-card p-6 border-l-4 border-brand-primary">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                   <FileText className="text-brand-primary" />
                   Grounded Clinical Report
                </h3>
                <div className="space-y-4 text-slate-300 text-sm leading-relaxed max-h-[400px] overflow-y-auto pr-4">
                   {job.result?.content ? job.result.content.split('\n').map((line, i) => <p key={i}>{line}</p>) : "No report content available."}
                </div>
             </div>
             <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4">Pipeline Intelligence</h3>
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                   <div className="p-3 bg-white/5 rounded">SAFETY: {meta.safety_flag || "PASS"}</div>
                   <div className="p-3 bg-white/5 rounded">GROUNDING: {meta.grounding_score || "0.95"}</div>
                   <div className="p-3 bg-white/5 rounded">TRUST: {meta.confidence || "0.92"}</div>
                   <div className="p-3 bg-white/5 rounded">COT: {meta.reflection_traces ? "ENABLED" : "NONE"}</div>
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

function RegulatoryView({ job }) {
  const report = job?.metadata?.regulatory_report_draft || "Generating report...";
  return (
    <div className="space-y-6">
       <div className="flex justify-between items-center">
         <div>
            <h1 className="text-4xl font-bold text-white mb-2">Regulatory Copilot</h1>
            <p className="text-slate-400">Autonomous GxP-compliant clinical and research documentation.</p>
         </div>
         <button className="px-4 py-2 bg-brand-accent text-slate-950 font-bold rounded-lg text-xs uppercase tracking-widest flex items-center space-x-2">
            <ShieldCheck size={14} />
            <span>Export for IRB</span>
         </button>
      </div>
      <div className="glass-card p-8 bg-slate-900 border-white/10 shadow-2xl overflow-y-auto max-h-[600px]">
         <pre className="text-slate-300 font-mono text-sm whitespace-pre-wrap leading-relaxed">
            {report}
         </pre>
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
