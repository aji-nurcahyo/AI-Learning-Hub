import levels from '../content/levels.json';
import tracks from '../content/specializations/tracks.json';
import gettingStartedModules from '../content/getting-started/modules.json';
export { levels, tracks, gettingStartedModules };

export type Resource = { status: 'unavailable' | 'available'; path: string | null };
export type Chapter = {
  id: number;
  slug: string;
  levelId: number;
  title: string;
  status: 'coming-soon' | 'published';
  description: string;
  objectives: string[];
  prerequisites: number[];
  tags: string[];
  resources: Record<string, Resource>;
  references: { title: string; url: string; status: string }[];
  validation: { reviewed: boolean; executionVerified: boolean };
  directory: string;
};

const entries = import.meta.glob('../content/**/metadata.json', { eager: true, import: 'default' }) as Record<string, Omit<Chapter, 'directory'>>;
export const chapters: Chapter[] = Object.entries(entries)
  .map(([path, c]) => ({ ...c, directory: path.replace('../', '').replace('/metadata.json', '') }))
  .sort((a, b) => a.id - b.id);

const text = import.meta.glob('../content/**/*.{md,json}', { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;

export const resourceText = (chapter: Chapter, key: string) =>
  text[`../${chapter.directory}/${chapter.resources[key]?.path}`] || '';

export const resourceUrl = (chapter: Chapter, key: string) =>
  `${import.meta.env.BASE_URL}${chapter.directory}/${chapter.resources[key].path}`;

export const gettingStartedText = (id: string) =>
  text[`../content/getting-started/${id}.md`] || '';

export const labels: Record<string, string> = {
  reading: 'Modul lengkap',
  slides: 'Slide interaktif',
  practice: 'Laboratorium & Praktik',
};
