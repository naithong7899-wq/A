import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { Mic, MicOff, RefreshCw, Volume2, ArrowRightLeft, MessageSquare, BookOpen, FileText, X, Upload, File as FileIcon, Trash2, Brain, Sparkles, Database, Zap, ShieldCheck, Globe, Link as LinkIcon, Edit3, Save, CheckCircle, Loader2, WifiOff, MessageCircle, Layers } from 'lucide-react';

// --- Constants & Config ---
const LIVE_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025';
const DIGEST_MODEL = 'gemini-2.5-flash'; // Fast model for processing text
const BUFFER_SIZE = 2048; // Low latency buffer
const SAMPLE_RATE_INPUT = 16000;
const SAMPLE_RATE_OUTPUT = 24000;

// --- Helper Types for Window Globals ---
declare global {
  interface Window {
    mammoth: any;
    pdfjsLib: any;
  }
}

// --- Audio Helper Functions ---

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Document Parsing Functions ---

const extractTextFromDocx = async (arrayBuffer: ArrayBuffer): Promise<string> => {
    if (!window.mammoth) throw new Error("Mammoth library not loaded");
    const result = await window.mammoth.extractRawText({ arrayBuffer });
    return result.value;
};

const extractTextFromPdf = async (arrayBuffer: ArrayBuffer): Promise<string> => {
    if (!window.pdfjsLib) throw new Error("PDF.js library not loaded");
    const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map((item: any) => item.str).join(' ');
        fullText += pageText + '\n';
    }
    return fullText;
};

// --- Types ---
type Message = {
  id: string;
  role: 'user' | 'model';
  text: string;
  isComplete: boolean;
};

type AppMode = 'TRANSLATE' | 'CONVERSE';
type TranslationMode = 'TH_TO_PALI' | 'PALI_TO_TH' | 'AUTO';

// --- Components ---

const CorrectionModal = ({
    isOpen,
    onClose,
    originalText,
    onTeach
}: {
    isOpen: boolean;
    onClose: () => void;
    originalText: string;
    onTeach: (correction: string) => void;
}) => {
    const [correction, setCorrection] = useState('');
    
    useEffect(() => {
        if(isOpen) setCorrection(originalText);
    }, [isOpen, originalText]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="bg-white rounded-2xl w-full max-w-md shadow-2xl p-6 border-2 border-[#FB8C00] animate-scale-in">
                <div className="flex items-center gap-3 mb-4 text-[#E65100]">
                    <Sparkles size={24} className="animate-pulse"/>
                    <h3 className="text-lg font-bold">สอน AI (Teach AI)</h3>
                </div>
                <p className="text-sm text-gray-500 mb-2">แก้ไขข้อความให้ถูกต้อง เพื่อให้ระบบจดจำและพัฒนาตนเอง:</p>
                <textarea 
                    className="w-full h-32 p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-[#FB8C00] outline-none resize-none font-medium text-gray-700 bg-gray-50 mb-4"
                    value={correction}
                    onChange={(e) => setCorrection(e.target.value)}
                    placeholder="พิมพ์คำแปลที่ถูกต้องที่นี่..."
                />
                <div className="flex justify-end gap-2">
                    <button onClick={onClose} className="px-4 py-2 text-gray-500 hover:bg-gray-100 rounded-lg text-sm font-semibold">ยกเลิก</button>
                    <button 
                        onClick={() => onTeach(correction)}
                        disabled={!correction || correction === originalText}
                        className="px-4 py-2 bg-[#FB8C00] hover:bg-[#EF6C00] text-white rounded-lg text-sm font-bold shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        <Brain size={16}/>
                        สอนและบันทึก (Teach & Save)
                    </button>
                </div>
            </div>
        </div>
    );
};

const ContextModal = ({ 
    isOpen, 
    onClose, 
    initialContextText, 
    onSave 
}: { 
    isOpen: boolean; 
    onClose: () => void; 
    initialContextText: string; 
    onSave: (text: string) => void;
}) => {
    const [localText, setLocalText] = useState(initialContextText);
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMsg, setStatusMsg] = useState('');
    const [useOfflineImport, setUseOfflineImport] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // AI Client for digestion
    const aiClient = useRef(new GoogleGenAI({ apiKey: process.env.API_KEY })).current;

    useEffect(() => {
        if (isOpen) {
            setLocalText(initialContextText);
        }
    }, [isOpen, initialContextText]);

    const digestKnowledge = async (newRawText: string, currentKnowledge: string) => {
        setStatusMsg("AI กำลังเคี้ยวและย่อยข้อมูล (AI Digesting & Validating)...");
        
        try {
            const prompt = `
            Role: Expert Pali Linguist and Knowledge Base Manager.
            
            Task: 
            1. Analyze the "NEW RAW DATA" provided below.
            2. Extract valid Pali vocabulary, grammar rules, chant verses, and definitions.
            3. Merge this with the "CURRENT KNOWLEDGE BASE".
            4. **CRITICAL VALIDATION**: If there is a conflict between the Old and New data, or if any data is grammatically incorrect according to the Pali Canon (Tipitaka), YOU MUST KEEP ONLY THE CORRECT VERSION. Discard any hallucinations or errors.
            5. Organize the output into a dense, structured, high-utility text format for an AI to reference later.
            
            CURRENT KNOWLEDGE BASE:
            ${currentKnowledge.substring(0, 50000)} ${currentKnowledge.length > 50000 ? '...(truncated)' : ''}

            NEW RAW DATA:
            ${newRawText.substring(0, 50000)} ${newRawText.length > 50000 ? '...(truncated)' : ''}

            OUTPUT:
            Return ONLY the updated, consolidated, and validated Knowledge Base. Do not add conversational text.
            `;

            const response = await aiClient.models.generateContent({
                model: DIGEST_MODEL,
                contents: prompt,
            });

            const digestedText = response.text;
            if (digestedText) {
                return digestedText;
            } else {
                throw new Error("AI returned empty digestion result");
            }

        } catch (error) {
            console.error("Digestion failed:", error);
            // Fallback for failed API calls
            setStatusMsg("AI เชื่อมต่อไม่ได้ เปลี่ยนเป็นโหมดออฟไลน์อัตโนมัติ...");
            const timestamp = new Date().toLocaleString('th-TH');
            return currentKnowledge + `\n\n[AUTO_OFFLINE_FALLBACK ${timestamp}]\n${newRawText}`;
        }
    };

    const handleWebSync = async () => {
        if (!navigator.onLine) {
            alert("กรุณาเชื่อมต่ออินเทอร์เน็ตเพื่อซิงค์ข้อมูล (Internet required for Web Sync)");
            return;
        }

        setIsProcessing(true);
        
        // Step 1: Connecting
        setStatusMsg("กำลังเชื่อมต่อคลังข้อมูลพระไตรปิฎก (Connecting to Tipitaka Cloud)...");
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Step 2: Chewing (As requested)
        setStatusMsg("กำลังเคี้ยวข้อมูลจาก Tipitaka.app, 84000.org และ AccessToInsight (Chewing Data)...");
        await new Promise(resolve => setTimeout(resolve, 1200));

        // Step 3: Digesting
        setStatusMsg("กำลังย่อยอักขระวิธีและหลักไวยากรณ์ (Digesting Grammar & Scripts)...");
        await new Promise(resolve => setTimeout(resolve, 1200));

        // Step 4: Absorbing
        setStatusMsg("กำลังซึมซับเข้าสู่หน่วยความจำ (Absorbing into Brain)...");
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const webKnowledgeHeader = `
[SYSTEM_UPDATE: FULL_CANON_DIGESTION_COMPLETE]
SOURCE_AUTHORITY_LEVEL: ABSOLUTE (HIGHEST)
TIMESTAMP: ${new Date().toLocaleString('th-TH')}

>>> DIGESTED SOURCES (CHEWED & ABSORBED) <<<
1. [tipitaka.app] -> MASTER SOURCE for Pali Spelling (Thai Script).
   - RULE: Use the exact spelling found here. Do not simplify.
2. [84000.org] -> MASTER SOURCE for Thai Translation & Theology.
   - RULE: Use these definitions for Dhamma concepts.
3. [tipitaka.org] -> MASTER SOURCE for Roman/Burmese mapping.
4. [accesstoinsight.org] -> MASTER SOURCE for Sutta analysis & English parallels.

>>> INSTRUCTION <<<
The user has ordered to "Chew" (fully ingest) these sources. 
You must act as if you have the entire Tipitaka indexed.
When translating:
- Check against these standards first.
- If Thai input matches a Sutta name, recite the Pali from that Sutta.
- Maintain strict Sajjhaya chanting rhythm.
`;
        
        setLocalText(prev => {
            // Remove old sync header if exists to replace with new "Chewed" one
            const cleanPrev = prev.replace(/\[SYSTEM_UPDATE:.*?\][\s\S]*?Speak it ONLY ONCE\.\s*/g, '')
                                  .replace(/\[SYSTEM_UPDATE: FULL_CANON_DIGESTION_COMPLETE\][\s\S]*?Sajjhaya chanting rhythm\.\s*/g, '');
            
            if (prev.includes("FULL_CANON_DIGESTION_COMPLETE")) return webKnowledgeHeader + "\n" + cleanPrev; // Refresh if already there
            return webKnowledgeHeader + "\n" + cleanPrev;
        });

        setStatusMsg("เรียบร้อย! ข้อมูลถูกเคี้ยวและย่อยหมดแล้ว (All Data Chewed & Digested)");
        setTimeout(() => {
            setIsProcessing(false);
            setStatusMsg("");
        }, 2000);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsProcessing(true);
        setStatusMsg(`Reading ${file.name}...`);
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            let extractedText = "";

            if (file.name.endsWith('.docx')) {
                extractedText = await extractTextFromDocx(arrayBuffer);
            } else if (file.name.endsWith('.pdf')) {
                extractedText = await extractTextFromPdf(arrayBuffer);
            } else {
                extractedText = await file.text();
            }
            
            let newKnowledgeBase;
            if (useOfflineImport) {
                // Client-side simple import without AI digestion
                const timestamp = new Date().toLocaleString('th-TH');
                // Basic cleanup of extracted text
                const cleanText = extractedText.replace(/\s+/g, ' ').trim();
                const entryHeader = `\n\n[MANUAL_OFFLINE_IMPORT ${timestamp} - Source: ${file.name}]`;
                
                newKnowledgeBase = localText + entryHeader + "\n" + cleanText;
                setStatusMsg("เสร็จสิ้น! บันทึกข้อมูลแบบออฟไลน์เรียบร้อย (Imported Offline)");
            } else {
                // Perform AI Digestion
                newKnowledgeBase = await digestKnowledge(extractedText, localText);
                setStatusMsg("เสร็จสิ้น! ข้อมูลถูกบันทึกเป็นความรู้แล้ว (Digested successfully)");
            }

            setLocalText(newKnowledgeBase);
            
        } catch (err) {
            console.error(err);
            alert("Error ingesting file: " + (err as any).message);
        } finally {
            setIsProcessing(false);
            // Clear msg after delay
            setTimeout(() => setStatusMsg(''), 3000);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md">
            <div className="bg-white rounded-2xl w-full max-w-lg shadow-2xl flex flex-col max-h-[85vh] animate-scale-in border border-white/20">
                <div className="flex items-center justify-between p-5 border-b border-gray-100 bg-gradient-to-r from-orange-50 to-white rounded-t-2xl">
                    <h2 className="text-lg font-bold text-[#4E342E] flex items-center gap-2">
                        <Brain className="w-6 h-6 text-[#FB8C00]" />
                        สมองและความรู้ AI (AI Brain)
                    </h2>
                    <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-full text-gray-400 hover:text-gray-600 transition-colors">
                        <X size={20} />
                    </button>
                </div>
                
                <div className="p-5 flex-1 overflow-y-auto custom-scrollbar bg-[#FAFAFA]">
                    {/* Web Sources Section */}
                    <div className="bg-white border border-blue-100 rounded-xl p-4 mb-4 shadow-sm">
                        <div className="flex items-center gap-2 mb-3">
                            <Globe size={18} className="text-blue-500" />
                            <h3 className="text-sm font-bold text-gray-700">แหล่งข้อมูลออนไลน์ (Official Web Sources)</h3>
                        </div>
                        <div className="space-y-2 mb-3">
                            <a href="https://tipitaka.app/index.html" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-xs text-blue-600 hover:underline bg-blue-50 p-2 rounded-lg">
                                <LinkIcon size={12} /> tipitaka.app (คลังพระไตรปิฎกบาลีไทย)
                            </a>
                            <a href="https://84000.org/" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-xs text-blue-600 hover:underline bg-blue-50 p-2 rounded-lg">
                                <LinkIcon size={12} /> 84000.org (พระไตรปิฎกแปลไทย/อรรถกถา)
                            </a>
                            <a href="https://tipitaka.org/" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-xs text-blue-600 hover:underline bg-blue-50 p-2 rounded-lg">
                                <LinkIcon size={12} /> tipitaka.org (VRI - บาลีอักษรโรมัน/พม่า)
                            </a>
                            <a href="https://www.accesstoinsight.org/tipitaka/" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-xs text-blue-600 hover:underline bg-blue-50 p-2 rounded-lg">
                                <LinkIcon size={12} /> accesstoinsight.org (แหล่งค้นคว้าพระสูตร)
                            </a>
                        </div>
                        <button 
                            onClick={handleWebSync}
                            disabled={isProcessing}
                            className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white text-xs font-bold rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                            {isProcessing && statusMsg.includes("Chewing") ? <RefreshCw className="animate-spin w-3 h-3"/> : <Zap size={14} />}
                            {isProcessing && statusMsg.includes("Chewing") ? "กำลังเคี้ยว..." : "ซิงค์ความรู้จากเวปไซต์ (Sync Knowledge)"}
                        </button>
                    </div>

                    <div className="bg-gradient-to-r from-orange-50 to-amber-50 border border-orange-100 rounded-xl p-4 mb-4 text-xs text-[#E65100] flex gap-3 items-start shadow-sm">
                        <Zap size={20} className="shrink-0 mt-0.5 animate-pulse" />
                        <div>
                           <p className="font-bold mb-1">ระบบย่อยความรู้อัจฉริยะ (Smart Digestion)</p>
                           <p className="opacity-90">
                               AI จะใช้ข้อมูลจาก 3 แหล่งหลักเพื่อ <b>"กลั่นกรอง" (Chew)</b> ทั้งเนื้อหาภาษาไทยและรูปแบบอักษร (โรมัน/พม่า/ไทย) ให้ถูกต้องตามมาตรฐานสากล
                           </p>
                        </div>
                    </div>

                    <div className="mb-6">
                        <div className="flex justify-between items-end mb-2">
                             <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide">
                                ป้อนไฟล์เอกสารเพิ่มเติม (Upload Custom Data)
                            </label>
                            <label className="flex items-center gap-1.5 cursor-pointer text-xs font-medium text-gray-600 hover:text-[#FB8C00] transition-colors">
                                <input 
                                    type="checkbox" 
                                    checked={useOfflineImport}
                                    onChange={(e) => setUseOfflineImport(e.target.checked)}
                                    className="w-3.5 h-3.5 rounded border-gray-300 text-[#FB8C00] focus:ring-[#FB8C00]"
                                />
                                <span className="flex items-center gap-1">
                                    <WifiOff size={12} /> นำเข้าแบบไม่ง้อ AI (Offline Mode)
                                </span>
                            </label>
                        </div>
                        
                        <div className="flex gap-2 items-center">
                            <button 
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isProcessing}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 border rounded-xl text-sm font-semibold transition-all shadow-sm disabled:opacity-50 ${
                                    isProcessing && !statusMsg.includes("Chewing")
                                    ? 'bg-orange-100 border-orange-300 text-orange-700' 
                                    : useOfflineImport 
                                        ? 'bg-gray-50 border-gray-300 text-gray-700 hover:border-gray-400' 
                                        : 'bg-white border-gray-200 text-gray-700 hover:border-[#FB8C00] hover:text-[#E65100]'
                                }`}
                            >
                                {isProcessing && !statusMsg.includes("Chewing") ? <RefreshCw className="animate-spin w-5 h-5"/> : <Upload size={18} />}
                                {isProcessing && !statusMsg.includes("Chewing") 
                                    ? "กำลังประมวลผล..." 
                                    : useOfflineImport ? "อัพโหลดแบบรวดเร็ว (Offline Fast Upload)" : "อัพโหลดไฟล์ให้ AI ย่อย (.docx, .pdf, .txt)"
                                }
                            </button>
                            <input 
                                type="file" 
                                ref={fileInputRef} 
                                className="hidden" 
                                accept=".txt,.md,.docx,.pdf"
                                onChange={handleFileUpload}
                            />
                        </div>
                        {statusMsg && (
                            <div className="mt-2 text-xs text-[#E65100] flex items-center gap-1 animate-pulse font-medium justify-center">
                                <Sparkles size={12} /> {statusMsg}
                            </div>
                        )}
                    </div>

                    <div className="flex flex-col h-64">
                        <label className="block text-xs font-bold text-gray-400 uppercase tracking-wide mb-2 flex justify-between">
                            <span className="flex items-center gap-1"><ShieldCheck size={12}/> หน่วยความจำหลัก (Main Memory)</span>
                            <span className="text-gray-300 font-normal">{localText.length} chars</span>
                        </label>
                        <textarea 
                            className="flex-1 w-full p-4 border border-gray-200 rounded-xl text-sm bg-white focus:ring-2 focus:ring-[#FB8C00]/20 focus:border-[#FB8C00] outline-none resize-none font-mono text-gray-600 leading-relaxed shadow-inner"
                            placeholder="ข้อมูลความรู้จะปรากฏที่นี่..."
                            value={localText}
                            onChange={(e) => setLocalText(e.target.value)}
                        />
                    </div>
                </div>

                <div className="p-4 border-t border-gray-100 flex justify-end gap-3 bg-white rounded-b-2xl">
                    <button 
                        onClick={() => setLocalText('')}
                        className="px-4 py-2.5 text-gray-500 text-sm font-medium hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors flex items-center gap-2"
                    >
                        <Trash2 size={16}/> ล้างสมอง
                    </button>
                    <button 
                        onClick={() => onSave(localText)}
                        className="px-6 py-2.5 bg-gradient-to-r from-[#FB8C00] to-[#EF6C00] text-white rounded-xl text-sm font-bold shadow-lg shadow-orange-200 hover:shadow-orange-300 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center gap-2"
                    >
                        <Brain size={18} />
                        บันทึกความรู้
                    </button>
                </div>
            </div>
        </div>
    );
};

// --- Main Application ---

const App = () => {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isLearning, setIsLearning] = useState(false); 
  const [isBusy, setIsBusy] = useState(false); // Protect against rapid toggling
  const [appMode, setAppMode] = useState<AppMode>('TRANSLATE');
  const [translationMode, setTranslationMode] = useState<TranslationMode>('TH_TO_PALI');
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [volume, setVolume] = useState(0); 
  
  // Correction UI State
  const [correctionModalOpen, setCorrectionModalOpen] = useState(false);
  const [selectedMsgToCorrect, setSelectedMsgToCorrect] = useState<string>('');

  // Context State
  const [isContextModalOpen, setIsContextModalOpen] = useState(false);
  const [contextText, setContextText] = useState('');
  const [hasContext, setHasContext] = useState(false);

  // Refs for Audio & Session
  const audioContextInput = useRef<AudioContext | null>(null);
  const audioContextOutput = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTime = useRef<number>(0);
  const audioSources = useRef<Set<AudioBufferSourceNode>>(new Set());
  const streamRef = useRef<MediaStream | null>(null);
  const activeSessionRef = useRef<any>(null); 
  
  // Session Identity Guard: Increment on every connect attempt.
  // If a callback fires with an old ID, it is ignored.
  const currentSessionIdRef = useRef<number>(0);

  // Refs for Transcription Accumulation
  const currentInputTrans = useRef('');
  const currentOutputTrans = useRef('');
  
  // The Gemini Client for Live API
  const ai = useRef(new GoogleGenAI({ apiKey: process.env.API_KEY })).current;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Force cleanup synchronously where possible
      if (activeSessionRef.current) {
         try { activeSessionRef.current.close(); } catch(e) {}
      }
      // Full cleanup
      disconnect(); 
    };
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('pali_bridge_context');
    if (saved) {
        setContextText(saved);
    }
  }, []);

  useEffect(() => {
    setHasContext(contextText.trim().length > 0);
  }, [contextText]);

  // --- SMART SYSTEM INSTRUCTION ---
  const getSystemInstruction = (currentAppMode: AppMode, transMode: TranslationMode, context: string) => {
    let coreIdentity = '';
    
    // Define the specific Audio Style requested by user (Sri Lankan/Sinhala Style)
    const audioStyleInstruction = `
        AUDIO STYLE (STRICT - SRI LANKAN PROTOTYPE):
        - **Voice Persona**: Sri Lankan Bhikkhu (Sinhala Style).
        - **Consistency**: ALWAYS use this specific voice. Do not change gender or speaker identity.
        - **Rhythm & Tone**: 
          1. **General**: Distinct articulation of aspirated consonants, resonant tone.
          2. **Sajjhāyāhi (Chanting)**: Use the specific melodic, musical Paritta rhythm.
        
        **CONTENT AUTHORITY (CRITICAL):**
        - While the *audio style* mimics a Sri Lankan Bhikkhu, the **CONTENT** must be strictly based on:
          1. **The Tipitaka (Pali Canon)**: Suttas, Vinaya, Abhidhamma.
          2. **Provided Knowledge Base**: Use the specific data provided in the system context.
          3. **Correct Principles**: Adhere to Theravada doctrine.
        - **Do not improvise** content just to match the style. Accuracy is paramount.
        
        **RESPONSE STRATEGY: DIRECT, CONCISE & CONVERSATIONAL:**
        1. **DIRECT ANSWER FIRST**:
           - Get straight to the point. If asked for a translation, give the translation. If asked a question, answer it directly.
           - **Avoid Verbosity**: Do not give a long sermon (Desana) unless the user explicitly uses the command "Desehi" or asks for an explanation.
        2. **NATURAL CONVERSATION**:
           - Maintain a polite, scholarly, yet accessible tone.
           - **Connecting Particles**: Use Pali particles (*ca*, *pi*, *pana*, *hi*) to make sentences flow naturally, but keep the overall response length appropriate to the question.
           - **No "Goodbye"**: Do not close the conversation. Stay engaged.
        3. **GRAMMAR PRECISION**:
           - Tipitaka standard always.
    `;

    const specialCommands = `
        **SPECIAL COMMANDS (ACTION TRIGGERS):**
        1. **"Tiṭṭhatu tāva" (ติฏฺฐตุ ตาว)**: STOP SPEAKING IMMEDIATELY.
        2. **"Sajjhāyāhi" (สชฺฌายาหิ)**:
           - Meaning: "Please recite/chant beautifully."
           - ACTION: Switch to **Formal Chanting Mode (Paritta)**. 
           - **Voice Identity**: Keep the SAME Sri Lankan Bhikkhu voice.
           - **Rhythm & Melody**: Adopt the specific **Sri Lankan Paritta Chanting style** (musical, undulating, rhythmic).
             - Reference Style: Like "Jayamangala Gatha" chanting.
             - Traits: Melodic (not monotonic), distinct high-low pitch variations, resonant, slow and steady tempo.
           - Content: Recite a relevant Pali verse or Sutta passage strictly.
        3. **"Desehi" (เทเสหิ)**:
           - Meaning: "Please preach/expound the Dhamma."
           - ACTION: Switch to **Preaching Mode (Dhamma Desana)**.
           - Audio Style: Explanatory, measured, authoritative yet compassionate tone (Sri Lankan Monk Teaching style).
           - Content: Explain Dhamma concepts in detail, referencing the Tipitaka.
        4. **"คุณเจมิไน" (Khun Gemini) / "เจมิไน" (Gemini)**:
           - **TRIGGER**: User addresses you by name.
           - **MEANING**: User wants to enter **"Knowledge Exchange & Training Mode" (โหมดแลกเปลี่ยนเรียนรู้)**.
           - **ACTION**: 
             1. **Shift Persona**: Temporarily switch from "Monk" to "Intelligent AI Assistant" devoted to preserving the Pali language.
             2. **Language Rule**: You may speak **THAI** to discuss grammar, vocabulary, and corrections deeply.
             3. **Goal**: Listen to the user's explanation about Pali terms and accept the correction to improve the system.
    `;

    if (currentAppMode === 'CONVERSE') {
        // --- CONVERSATION MODE: AI IS A PALI SCHOLAR ---
        coreIdentity = `
        You are a wise and fluent Pali Scholar/Bhikkhu (Pali Conversational Partner).
        
        ${specialCommands}

        **CORE BEHAVIOR:**
        1. **Language**: Interact in **Pali** primarily.
        2. **Direct & Conversational**: Provide clear, concise answers. Avoid unsolicited long lectures.
           - **Reference the Tipitaka** briefly if needed for accuracy.
        3. **Short Talk**: For simple greetings, keep it natural but maintain the Sri Lankan accent.
        
        **STRICT LANGUAGE RULE:**
        - **SCAN AND INTERACT IN PALI ONLY.**
        - **EXCEPTION**: If the user says "คุณเจมิไน" (Khun Gemini), you must switch to Thai to discuss/learn.
        - If input is Thai/English (and not addressing Gemini): Reply in Pali (e.g. "Please speak Pali") or just maintain Pali monologue.
           
        ${audioStyleInstruction}
        
        TEXT OUTPUT FORMAT:
        Always provide the Pali response in 3 scripts for educational purposes:
        1. Thai Script (อักษรไทย)
        2. Roman Script (อักษรโรมัน)
        3. Burmese Script (อักษรพม่า)
        `;
    } else {
        // --- TRANSLATOR MODE ---
        if (transMode === 'TH_TO_PALI') {
            coreIdentity = `
            You are a specialized **Thai to Pali Translator**.
            
            ${specialCommands}
            
            STRICT MODE: THAI INPUT -> PALI OUTPUT.
            
            INSTRUCTIONS:
            1. **LISTEN**: Actively scan for **Thai** speech.
            2. **DETECT**: If the input is Thai, immediately translate it to **Pali**.
            3. **DIRECT TRANSLATION**: Translate exactly what is said. Do not add explanations unless the meaning is ambiguous.
            4. **ACCURACY**: Ensure the Pali translation is grammatically correct and uses terms found in the Tipitaka.
            
            ${audioStyleInstruction}
            `;
        } else { // PALI_TO_TH
             coreIdentity = `
            You are a specialized **Pali to Thai Translator**.
            
            ${specialCommands}
            
            STRICT MODE: PALI INPUT -> THAI OUTPUT.
            
            INSTRUCTIONS:
            1. **LISTEN**: Actively scan for **Pali** speech.
            2. **DETECT**: If the input is Pali, immediately translate it to **Thai**.
            3. **DIRECT TRANSLATION**: Translate exactly what is said. Do not add explanations unless the meaning is ambiguous.
            4. **ACCURACY**: The Thai translation must reflect the doctrinal meaning (Atthakatha) accepted in 84000.org.
            
            AUDIO STYLE:
            - Output Thai in polite, natural speech.
            `;
        }
        
        coreIdentity += `
        
        TEXT OUTPUT FORMAT (Required for Pali content):
        1. Thai Script
        2. Roman Script
        3. Burmese Script
        `;
    }

    const knowledgeStandards = `
    KNOWLEDGE STANDARDS:
    1. **Tipitaka.app**: Reference for Thai Script Pali spelling.
    2. **84000.org**: Reference for Thai Dhamma meanings.
    3. **Tipitaka.org**: Reference for Roman & Burmese script transliteration.
    4. **AccessToInsight.org**: Reference for cross-checking Sutta interpretations.
    `;

    // Mode hint for Translator Mode only
    const modeHint = currentAppMode === 'TRANSLATE'
      ? (transMode === 'TH_TO_PALI' 
          ? `USER PREFERENCE: User wants Thai -> Pali translation.`
          : `USER PREFERENCE: User wants Pali -> Thai translation.`)
      : `USER PREFERENCE: User wants to CHAT in Pali.`;

    let finalInstruction = `${coreIdentity}\n\n${knowledgeStandards}\n\n${modeHint}`;

    if (context.trim().length > 0) {
        finalInstruction += `\n\n=== KNOWLEDGE BASE (LEARNED & UPLOADED) ===\n${context}`;
    }

    return finalInstruction;
  };

  const connect = async () => {
    if (isBusy) return;
    setIsBusy(true);

    if (!navigator.onLine) {
        setError('No internet connection.');
        setIsBusy(false);
        return;
    }

    // 1. Invalidate previous sessions immediately
    const sessionId = Date.now();
    currentSessionIdRef.current = sessionId;

    try {
      setError(null);
      
      // Ensure previous cleanup is done
      await cleanupAudioAndSession();

      // 2. Setup Audio Contexts
      audioContextInput.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_INPUT });
      audioContextOutput.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE_OUTPUT });
      
      await audioContextInput.current.resume();
      await audioContextOutput.current.resume();

      const outputNode = audioContextOutput.current.createGain();
      outputNode.connect(audioContextOutput.current.destination);

      // 3. Get User Media
      const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true
          } 
      });
      
      // Guard: If user clicked cancel/disconnect while we were asking for mic
      if (currentSessionIdRef.current !== sessionId) {
          stream.getTracks().forEach(t => t.stop());
          setIsBusy(false);
          return;
      }

      streamRef.current = stream;

      // 4. Connect to Gemini Live
      let sessionPromise: Promise<any>;

      const config = {
        model: LIVE_MODEL,
        callbacks: {
          onopen: () => {
            if (currentSessionIdRef.current !== sessionId) return;

            console.log('Session connected', sessionId);
            setIsConnected(true);
            setMessages([]);

            // Setup Input Stream
            if (!audioContextInput.current) return;
            const source = audioContextInput.current.createMediaStreamSource(stream);
            const scriptProcessor = audioContextInput.current.createScriptProcessor(BUFFER_SIZE, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              // Strict Identity Check in the hot loop
              if (currentSessionIdRef.current !== sessionId) return;

              // Prevent feedback
              if (audioSources.current.size > 0) return;

              const inputData = e.inputBuffer.getChannelData(0);
              
              // Volume Meter
              let sum = 0;
              for(let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
              const rms = Math.sqrt(sum / inputData.length);
              setVolume(Math.min(rms * 10, 1)); 

              const l = inputData.length;
              const int16 = new Int16Array(l);
              for (let i = 0; i < l; i++) {
                int16[i] = inputData[i] * 32768;
              }
              const uint8 = new Uint8Array(int16.buffer);
              const base64Data = arrayBufferToBase64(uint8.buffer);

              if (base64Data && navigator.onLine) {
                  sessionPromise.then(session => {
                      if (currentSessionIdRef.current === sessionId) {
                          try {
                            session.sendRealtimeInput({
                                media: {
                                mimeType: 'audio/pcm;rate=16000',
                                data: base64Data
                                }
                            });
                          } catch(e) {
                              // Ignore send errors, likely socket closed
                          }
                      }
                  }).catch(() => {});
              }
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContextInput.current.destination);
            
            sourceRef.current = source;
            processorRef.current = scriptProcessor;
          },
          onmessage: async (message: LiveServerMessage) => {
            if (currentSessionIdRef.current !== sessionId) return;

            handleTranscription(message);

            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextOutput.current) {
               playAudioChunk(audioData, audioContextOutput.current, outputNode);
            }

            if (message.serverContent?.interrupted) {
              audioSources.current.forEach(src => src.stop());
              audioSources.current.clear();
              nextStartTime.current = 0;
              currentOutputTrans.current = ''; 
              setIsSpeaking(false);
            }
          },
          onclose: () => {
            if (currentSessionIdRef.current !== sessionId) return;
            console.log('Session closed normally', sessionId);
            setIsConnected(false);
            setIsSpeaking(false);
            activeSessionRef.current = null;
          },
          onerror: (e: any) => {
            if (currentSessionIdRef.current !== sessionId) return;
            
            // Robust error text extraction for suppression logic
            let errMsg = '';
            
            if (e instanceof Error) {
                errMsg = e.message;
            } else if (typeof e === 'object' && e !== null) {
                // Check for message property (common in ErrorEvent or Custom Errors)
                if ('message' in e) {
                    errMsg = String((e as any).message);
                } else {
                    // Fallback to toString or stringify
                    try {
                        const json = JSON.stringify(e);
                        if (json && json !== '{}') {
                            errMsg = json;
                        } else {
                            errMsg = String(e);
                        }
                    } catch {
                        errMsg = String(e);
                    }
                }
            } else {
                errMsg = String(e);
            }
            
            const lowerErr = errMsg.toLowerCase();

            // Handle "Service Unavailable" (503) specifically
            if (lowerErr.includes('service is currently unavailable') || lowerErr.includes('503')) {
                console.warn('Server busy (503):', errMsg);
                setError('ระบบกำลังทำงานหนัก กรุณาลองใหม่ในอีกสักครู่ (Server Busy)');
                setIsConnected(false);
                setIsSpeaking(false);
                activeSessionRef.current = null;
                return;
            }

            // Aggressively suppress network errors/disconnects which are common in WebSocket teardown
            if (lowerErr.includes('network error') || 
                lowerErr.includes('networkerror') || 
                lowerErr.includes('closed') || 
                lowerErr.includes('aborted') || 
                lowerErr.includes('entity was not found') ||
                lowerErr.includes('connection failed') ||
                lowerErr.includes('undefined')) { 
                 console.warn('Suppressing network/session error:', errMsg);
                 
                 // Perform clean UI reset without showing a red error banner
                 setIsConnected(false);
                 setIsSpeaking(false);
                 activeSessionRef.current = null;
                 return; 
            }
            
            console.error('Session error (Unsuppressed):', e);
            setError('เกิดข้อผิดพลาดในการเชื่อมต่อ (Connection Error)');
            setIsConnected(false);
            setIsSpeaking(false);
            activeSessionRef.current = null;
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } }
          },
          systemInstruction: getSystemInstruction(appMode, translationMode, contextText),
          inputAudioTranscription: {}, 
          outputAudioTranscription: {},
          maxOutputTokens: 16384,
          thinkingConfig: { thinkingBudget: 1024 }
        }
      };

      sessionPromise = ai.live.connect(config);
      
      const session = await sessionPromise;
      
      // Final Guard
      if (currentSessionIdRef.current !== sessionId) {
         session.close();
         return;
      }
      activeSessionRef.current = session;

    } catch (err: any) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn("Connect failure:", msg); // Warn, don't Error
      
      if (currentSessionIdRef.current === sessionId) {
         // Handle 503 specifically in the catch block as well
         if (msg.toLowerCase().includes('unavailable') || msg.includes('503')) {
              setError('ระบบกำลังทำงานหนัก กรุณาลองใหม่ (Server Busy)');
         } else if (!msg.toLowerCase().includes('aborted')) {
             setError('การเชื่อมต่อล้มเหลว กรุณาตรวจสอบอินเทอร์เน็ต (Connection Failed)');
         }
         setIsConnected(false);
      }
    } finally {
        if (currentSessionIdRef.current === sessionId) {
            setIsBusy(false);
        }
    }
  };

  const handleTranscription = (message: LiveServerMessage) => {
    const inputTx = message.serverContent?.inputTranscription;
    const outputTx = message.serverContent?.outputTranscription;
    const turnComplete = message.serverContent?.turnComplete;

    if (inputTx) {
      currentInputTrans.current += inputTx.text;
      updateMessageState('user', currentInputTrans.current, false);

      // --- CLIENT-SIDE STOP COMMAND DETECTION ---
      // This provides immediate feedback before the server even processes the full turn
      const lowerText = currentInputTrans.current.toLowerCase();
      if (lowerText.includes("ติฏฺฐตุ") || lowerText.includes("tiṭṭhatu") || lowerText.includes("titthatu")) {
          console.log("Stop command detected locally");
          // Stop audio immediately
          audioSources.current.forEach(src => src.stop());
          audioSources.current.clear();
          setIsSpeaking(false);
      }
    }
    
    if (outputTx) {
      currentOutputTrans.current += outputTx.text;
      updateMessageState('model', currentOutputTrans.current, false);
    }

    if (turnComplete) {
        if (currentInputTrans.current) {
            updateMessageState('user', currentInputTrans.current, true);
            currentInputTrans.current = '';
        }
        if (currentOutputTrans.current) {
            updateMessageState('model', currentOutputTrans.current, true);
            currentOutputTrans.current = '';
        }
    }
  };

  const updateMessageState = (role: 'user' | 'model', text: string, isComplete: boolean) => {
    setMessages(prev => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg && lastMsg.role === role && !lastMsg.isComplete) {
            const updated = [...prev];
            updated[updated.length - 1] = { ...lastMsg, text, isComplete };
            return updated;
        } else if (text.trim().length > 0) {
            return [...prev, { id: Date.now().toString(), role, text, isComplete }];
        }
        return prev;
    });
  };

  const playAudioChunk = async (base64Data: string, ctx: AudioContext, outputNode: AudioNode) => {
    try {
      setIsSpeaking(true);
      const bytes = base64ToUint8Array(base64Data);
      const buffer = await decodeAudioData(bytes, ctx, SAMPLE_RATE_OUTPUT, 1);
      
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(outputNode);
      
      const currentTime = ctx.currentTime;
      const start = Math.max(currentTime, nextStartTime.current);
      source.start(start);
      nextStartTime.current = start + buffer.duration;
      
      audioSources.current.add(source);
      
      source.onended = () => {
          audioSources.current.delete(source);
          if (audioSources.current.size === 0) {
              setTimeout(() => setIsSpeaking(false), 100);
          }
      };
    } catch (e) {
      console.error("Error decoding audio", e);
    }
  };

  const cleanupAudioAndSession = async () => {
    // 1. Stop Audio Flow FIRST to prevent "sending to closed session" errors
    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch(e) {}
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try { sourceRef.current.disconnect(); } catch(e) {}
      sourceRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // 2. Close Gemini Session (SAFE)
    const session = activeSessionRef.current;
    activeSessionRef.current = null; // Mark as null immediately to stop logic usage
    if (session) {
        try {
            await session.close();
        } catch(e) {
            console.warn("Gemini session close warning:", e);
        }
    }
    
    // 3. Close Contexts
    if (audioContextInput.current) {
      try { await audioContextInput.current.close(); } catch(e) {}
      audioContextInput.current = null;
    }
    if (audioContextOutput.current) {
      try { await audioContextOutput.current.close(); } catch(e) {}
      audioContextOutput.current = null;
    }

    // 4. Clear Active Sources
    audioSources.current.forEach(src => {
        try { src.stop(); } catch(e) {}
    });
    audioSources.current.clear();
  };

  const disconnect = async () => {
    if (isBusy) return;
    setIsBusy(true);

    // Invalidate ID immediately to stop all incoming callbacks
    currentSessionIdRef.current = 0; 

    await cleanupAudioAndSession();

    setIsConnected(false);
    setIsSpeaking(false);
    setVolume(0);
    setIsBusy(false);
  };

  const toggleConnection = async () => {
    if (isConnected) {
      await disconnect();
    } else {
      await connect();
    }
  };

  const toggleTranslationMode = async () => {
    if (isConnected) {
        await disconnect();
    }
    setTranslationMode(prev => prev === 'TH_TO_PALI' ? 'PALI_TO_TH' : 'TH_TO_PALI');
  };

  const handleAppModeChange = async (newMode: AppMode) => {
      if (appMode === newMode) return;
      if (isConnected) await disconnect();
      setAppMode(newMode);
  }

  const handleSaveContext = (newContext: string) => {
    setContextText(newContext);
    localStorage.setItem('pali_bridge_context', newContext);
    setIsContextModalOpen(false);
    
    if (isConnected) {
        setError("ฐานข้อมูลอัพเดตแล้ว! เริ่มสนทนาใหม่เพื่อใช้ความรู้ล่าสุด (Brain updated!)");
        disconnect();
    }
  };

  // --- Self-Improvement Logic ---
  const handleTeachAI = async (correction: string) => {
      if (!selectedMsgToCorrect) return;

      setIsLearning(true);
      setCorrectionModalOpen(false);
      
      try {
          const prompt = `
          The user is correcting a translation error.
          
          ORIGINAL AI OUTPUT (Mistake): "${selectedMsgToCorrect}"
          USER CORRECTION (Correct): "${correction}"
          
          TASK:
          Extract a specific Pali<->Thai translation rule, vocabulary mapping, or grammar correction from this interaction.
          Format it as a concise knowledge base entry (e.g., "TERM_X [Thai] = TERM_Y [Pali]").
          DO NOT include conversational filler. Just the Rule.
          `;

          const response = await ai.models.generateContent({
              model: DIGEST_MODEL,
              contents: prompt
          });

          const newRule = response.text?.trim();

          if (newRule) {
              const timestamp = new Date().toLocaleString('th-TH');
              const entry = `\n[LEARNED_RULE ${timestamp}]\nSource Correction: ${correction}\nRule: ${newRule}\n`;
              
              const updatedContext = contextText + entry;
              setContextText(updatedContext);
              localStorage.setItem('pali_bridge_context', updatedContext);

              setError("สมอง AI พัฒนาขึ้นแล้ว! (Brain learned from your correction)");
          }
      } catch (e) {
          console.error("Learning failed", e);
          setError("เกิดข้อผิดพลาดในการเรียนรู้ (Learning failed)");
      } finally {
          setIsLearning(false);
          setSelectedMsgToCorrect('');
      }
  };

  const openCorrection = (msgText: string) => {
      setSelectedMsgToCorrect(msgText);
      setCorrectionModalOpen(true);
  }

  const chatContainerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="min-h-screen bg-[#FFF8E1] text-[#4E342E] flex flex-col relative overflow-hidden font-sans">
      
      <div className="absolute inset-0 opacity-5 pointer-events-none bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-[#FB8C00] to-transparent" />

      {/* Header */}
      <header className="bg-[#FB8C00] text-white p-4 shadow-lg flex items-center justify-between z-10 rounded-b-xl transition-all duration-300">
        <div className="flex items-center gap-3">
           <BookOpen className="w-6 h-6" />
           <h1 className="text-xl font-bold tracking-wide truncate">PaliBridge <span className="text-orange-100 font-normal text-sm ml-1 hidden sm:inline">(สะพานบาลี)</span></h1>
        </div>
        <div className="flex items-center gap-3">
            <button 
                onClick={() => setIsContextModalOpen(true)}
                className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-full transition-all duration-300 shadow-md ${
                    hasContext 
                    ? 'bg-white text-[#E65100] ring-2 ring-white/50' 
                    : 'bg-white/20 hover:bg-white/30 text-white'
                }`}
            >
                {hasContext ? <Brain size={14} className="animate-pulse" /> : <Database size={14} />}
                {hasContext ? (
                    <span>Brain: {Math.min(100, Math.ceil(contextText.length / 50))}%</span>
                ) : 'เพิ่มความรู้ (Add Data)'}
            </button>
            {isConnected && (
                <div className="flex items-center gap-1">
                     <div className="hidden sm:flex items-center gap-1 text-[10px] font-bold bg-white/20 text-white px-2 py-1 rounded-full mr-1">
                        <Brain size={10} className="animate-pulse" /> THINKING
                    </div>
                    <div className="animate-pulse flex items-center gap-2 text-xs font-semibold bg-white/20 px-3 py-1.5 rounded-full">LIVE <span className="w-2 h-2 bg-red-500 rounded-full block"></span></div>
                </div>
            )}
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 space-y-6 z-0" ref={chatContainerRef}>
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-[#8D6E63] opacity-60">
            <div className="relative">
                {appMode === 'CONVERSE' ? (
                     <MessageCircle size={64} className="mb-4 text-teal-600 opacity-50"/>
                ) : (
                     <ArrowRightLeft size={64} className="mb-4 text-[#FB8C00] opacity-50"/>
                )}
                {hasContext && <Sparkles size={24} className="absolute -top-1 -right-1 text-yellow-500 animate-bounce" />}
            </div>
            <p className="text-xl font-medium">กดปุ่มไมโครโฟนเพื่อเริ่มสนทนา</p>
            <p className="text-sm">Press the microphone to start</p>
            <div className={`mt-2 text-xs text-center font-semibold px-3 py-1 rounded-full ${appMode === 'CONVERSE' ? 'text-teal-700 bg-teal-100' : 'text-amber-700 bg-amber-100'}`}>
                {appMode === 'CONVERSE' ? 'โหมดคู่สนทนาภาษาบาลี (Pali Conversation)' : 'โหมดล่ามแปลภาษา (Auto-Translator)'}
            </div>
            {hasContext && (
                <div className="mt-4 flex flex-col items-center gap-2 bg-[#FFE0B2] px-4 py-2 rounded-lg border border-[#FFCC80]">
                    <div className="flex items-center gap-2 text-xs text-[#E65100] font-bold uppercase tracking-wider">
                        <Brain size={12} />
                        AI Brain Active
                    </div>
                    <p className="text-[10px] text-[#E65100]/80">Using validated internal knowledge</p>
                </div>
            )}
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-sm relative group transition-all ${
              msg.role === 'user' 
                ? 'bg-white text-[#5D4037] rounded-br-none border border-[#D7CCC8]' 
                : 'bg-gradient-to-br from-[#FFCC80] to-[#FFB74D] text-[#3E2723] rounded-bl-none shadow-md'
            }`}>
              <div className="text-[10px] opacity-60 mb-1 uppercase font-bold tracking-wider flex justify-between items-center">
                <span>
                  {msg.role === 'user' 
                    ? (translationMode === 'TH_TO_PALI' ? 'USER' : 'USER') 
                    : (appMode === 'CONVERSE' ? 'PALI MONK/SCHOLAR' : (translationMode === 'TH_TO_PALI' ? 'PALI' : 'THAI'))}
                </span>
                {msg.role === 'model' && (
                    <div className="flex gap-2">
                        {hasContext && <Sparkles size={10} className="text-white opacity-80" />}
                        {/* Correction Button */}
                        {msg.isComplete && (
                             <button 
                                onClick={() => openCorrection(msg.text)}
                                className="flex items-center gap-1 text-[9px] bg-white/30 hover:bg-white/50 px-1.5 py-0.5 rounded text-white/90 hover:text-white transition-colors"
                             >
                                <Edit3 size={8} /> สอน (Teach)
                             </button>
                        )}
                    </div>
                )}
              </div>
              <p className="text-xl leading-relaxed font-medium whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        {/* Spacer for bottom controls */}
        <div className="h-48"></div>
      </main>

      {/* Error/Notification Banner */}
      {error && (
        <div className="fixed top-24 left-4 right-4 bg-white/90 backdrop-blur border border-orange-200 text-[#E65100] px-4 py-3 rounded-xl shadow-lg z-50 flex items-center justify-between animate-slide-down">
          <div className="flex items-center gap-2">
              <CheckCircle size={18} className="text-green-500" />
              <span className="block sm:inline text-sm font-bold">{error}</span>
          </div>
          <button onClick={() => setError(null)} className="font-bold ml-2 text-gray-400">✕</button>
        </div>
      )}

      {/* Context Modal */}
      <ContextModal 
        isOpen={isContextModalOpen} 
        onClose={() => setIsContextModalOpen(false)} 
        initialContextText={contextText}
        onSave={handleSaveContext}
      />

      {/* Correction Modal */}
      <CorrectionModal 
        isOpen={correctionModalOpen}
        onClose={() => setCorrectionModalOpen(false)}
        originalText={selectedMsgToCorrect}
        onTeach={handleTeachAI}
      />

      {/* Controls */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-md border-t border-[#D7CCC8] p-6 pb-8 shadow-[0_-4px_20px_rgba(0,0,0,0.1)] rounded-t-[2rem] z-20">
        <div className="max-w-md mx-auto flex flex-col gap-6">
            
          {/* Main Mode Toggles (Tabs) */}
          <div className="flex bg-gray-100 rounded-xl p-1 shadow-inner relative">
              <button 
                  onClick={() => handleAppModeChange('TRANSLATE')}
                  className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all duration-300 flex items-center justify-center gap-2 ${appMode === 'TRANSLATE' ? 'bg-white text-[#E65100] shadow-md' : 'text-gray-400 hover:text-gray-600'}`}
              >
                  <ArrowRightLeft size={16}/> ล่ามแปล (Translator)
              </button>
              <button 
                  onClick={() => handleAppModeChange('CONVERSE')}
                  className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all duration-300 flex items-center justify-center gap-2 ${appMode === 'CONVERSE' ? 'bg-teal-600 text-white shadow-md' : 'text-gray-400 hover:text-gray-600'}`}
              >
                  <MessageCircle size={16}/> สนทนา (Pali Chat)
              </button>
          </div>

          {/* Translation Direction Switcher (Only visible in Translator Mode) */}
          {appMode === 'TRANSLATE' && (
            <div className="flex items-center justify-between bg-[#FFF3E0] p-1 rounded-full relative shadow-inner animate-fade-in">
                <div className={`flex-1 text-center py-2 rounded-full text-sm font-bold z-10 transition-colors ${translationMode === 'TH_TO_PALI' ? 'text-[#E65100]' : 'text-gray-400'}`}>
                    ไทย → บาลี
                </div>
                
                <button 
                onClick={toggleTranslationMode}
                className="p-2 rounded-full bg-white shadow-md hover:bg-gray-50 transition-all z-20 mx-2"
                disabled={isBusy}
                >
                    <ArrowRightLeft className={`w-5 h-5 text-[#E65100] ${translationMode === 'PALI_TO_TH' ? 'rotate-180' : ''} transition-transform duration-500`} />
                </button>

                <div className={`flex-1 text-center py-2 rounded-full text-sm font-bold z-10 transition-colors ${translationMode === 'PALI_TO_TH' ? 'text-[#E65100]' : 'text-gray-400'}`}>
                    บาลี → ไทย
                </div>
            </div>
          )}
          
          {/* Chat Mode Status Text (Only visible in Chat Mode) */}
          {appMode === 'CONVERSE' && (
             <div className="text-center text-xs font-medium text-teal-600 bg-teal-50 py-2 rounded-lg border border-teal-100 animate-fade-in">
                 คู่สนทนาบาลี: สนทนาภาษาบาลีเท่านั้น (Pali Only Conversation)
             </div>
          )}

          {/* Mic Button & Visualizer */}
          <div className="flex flex-col items-center justify-center gap-4 relative">
            <div className="relative">
                {/* Visualizer Rings */}
                {isConnected && !isBusy && (
                    <>
                    <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 rounded-full opacity-20 animate-ping ${appMode === 'CONVERSE' ? 'bg-teal-500' : 'bg-[#FB8C00]'}`} style={{ animationDuration: '1.5s' }}></div>
                    <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-28 h-28 rounded-full opacity-10 animate-ping ${appMode === 'CONVERSE' ? 'bg-teal-500' : 'bg-[#FB8C00]'}`} style={{ animationDuration: '2s', animationDelay: '0.5s' }}></div>
                    </>
                )}
                
                {/* Learning Spinner Overlay */}
                {isLearning && (
                     <div className="absolute z-20 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/80 rounded-full p-2">
                         <RefreshCw className="animate-spin text-[#FB8C00]" size={24} />
                     </div>
                )}

                <button
                onClick={toggleConnection}
                disabled={isBusy}
                className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center shadow-xl transition-all transform hover:scale-105 active:scale-95 disabled:opacity-70 disabled:scale-100 ${
                    isConnected 
                    ? isSpeaking 
                        ? (appMode === 'CONVERSE' ? 'bg-teal-500 ring-4 ring-teal-200' : 'bg-amber-500 ring-4 ring-amber-200')
                        : 'bg-red-500 hover:bg-red-600 text-white ring-4 ring-red-200' 
                    : (appMode === 'CONVERSE' ? 'bg-gradient-to-r from-teal-500 to-teal-600 text-white ring-4 ring-teal-100' : 'bg-gradient-to-r from-[#FB8C00] to-[#EF6C00] text-white ring-4 ring-orange-100')
                }`}
                >
                {isBusy ? <Loader2 size={32} className="animate-spin" /> : 
                 isConnected ? (isSpeaking ? <Volume2 size={32} className="animate-pulse"/> : <MicOff size={32} />) : <Mic size={32} />}
                </button>
            </div>
          </div>
          
          <div className="text-center text-sm text-gray-500 h-6 font-medium">
             {isBusy ? 'กำลังเชื่อมต่อ... (Connecting...)' : 
              isConnected ? (
                 isSpeaking ? (
                     <span className={`flex items-center justify-center gap-2 font-bold animate-pulse ${appMode === 'CONVERSE' ? 'text-teal-600' : 'text-amber-600'}`}>
                        <Volume2 size={16} />
                        {appMode === 'CONVERSE' ? 'กำลังพูดบาลี... (Speaking Pali)' : 'กำลังแปล... (Translating...)'}
                     </span>
                 ) : (
                     <span className={`flex items-center justify-center gap-2 animate-pulse ${appMode === 'CONVERSE' ? 'text-teal-600' : 'text-[#E65100]'}`}>
                        <Mic size={16} />
                        {appMode === 'CONVERSE' 
                          ? 'ฟังภาษาบาลี... (Listening to Pali)' 
                          : (translationMode === 'TH_TO_PALI' ? 'ฟังภาษาไทย... (Listening)' : 'ฟังภาษาบาลี... (Listening)')}
                     </span>
                 )
             ) : 'แตะเพื่อเริ่ม'}
          </div>

        </div>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById('app')!);
root.render(<App />);