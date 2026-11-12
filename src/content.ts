import levels from '../content/levels.json';
import tracks from '../content/specializations/tracks.json';
import gettingStartedModules from '../content/getting-started/modules.json';
export { levels, tracks, gettingStartedModules };

export type Resource = { status: 'unavailable' | 'available'; path: string | null };
export type ReferenceItem = { title: string; url: string; status: string };

export type Chapter = {
  id: number;
  slug: string;
  levelId: number;
  title: string;
  status: 'coming-soon' | 'published';
  description: string;
  highlights?: string[];
  objectives: string[];
  prerequisites: number[];
  tags: string[];
  resources: Record<string, Resource>;
  references: ReferenceItem[];
  validation: { reviewed: boolean; executionVerified: boolean };
  directory: string;
};

export type TrackLab = {
  title: string;
  chapterId: number;
  scriptPaths: string[];
};

export type TrackDetail = {
  id: string;
  code: string;
  slug: string;
  title: string;
  status: 'coming-soon' | 'published';
  focus: string;
  shortDescription: string;
  topics: string[];
  learningOutcomes: string[];
  prerequisites: number[];
  roadmap: {
    foundation: number[];
    core: number[];
    advanced: number[];
  };
  labs: TrackLab[];
  capstone: {
    title: string;
    summary: string;
    rubricTotal: number;
  };
  references: ReferenceItem[];
};

const entries = import.meta.glob('../content/level-*/**/metadata.json', { eager: true, import: 'default' }) as Record<string, Omit<Chapter, 'directory'>>;
export const chapters: Chapter[] = Object.entries(entries)
  .map(([path, c]) => ({ ...c, directory: path.replace('../', '').replace('/metadata.json', '') }))
  .sort((a, b) => a.id - b.id);

const trackEntries = import.meta.glob('../content/tracks/*/metadata.json', { eager: true, import: 'default' }) as Record<string, TrackDetail>;
export const trackDetails: Record<string, TrackDetail> = Object.fromEntries(
  Object.values(trackEntries).map(t => [t.id, t])
);

const text = import.meta.glob('../content/**/*.{md,json}', { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;

export const resourceText = (chapter: Chapter, key: string) =>
  text[`../${chapter.directory}/${chapter.resources[key]?.path}`] || '';

export const resourceUrl = (chapter: Chapter, key: string) =>
  `${import.meta.env.BASE_URL}${chapter.directory}/${chapter.resources[key].path}`;

export const chapterNotebookUrl = (chapter: Chapter, base: string) =>
  `${base}${chapter.directory}/notebooks/chapter_${String(chapter.id).padStart(2, '0')}_companion.ipynb`;

export const trackText = (trackId: string, section: 'overview' | 'practice' | 'capstone') =>
  text[`../content/tracks/${trackId}/${section}.md`] || '';

export const gettingStartedText = (id: string) =>
  text[`../content/getting-started/${id}.md`] || '';

export const labels: Record<string, string> = {
  reading: 'Modul lengkap',
  slides: 'Slide perkuliahan',
  practice: 'Laboratorium & Praktik',
};
