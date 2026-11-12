export interface SlideColumn {
  title: string;
  body?: string;
  items?: string[];
}

export interface SlideCaseBox {
  title?: string;
  scenario: string;
  goal?: string;
  options?: string[];
}

export interface SlideData {
  title: string;
  body: string;
  kind: 'cover' | 'content' | 'split' | 'code' | 'case' | 'summary' | string;
  notes: string;
  code?: string;
  tex?: string;
  diagram?: string[];
  columns?: SlideColumn[];
  caseBox?: SlideCaseBox;
  questions?: string[];
}

import template from '../../content/templates/slides.json';
export const templateSlides = (title: string): SlideData[] =>
  (template as SlideData[]).map(s => ({ ...s, title: s.kind === 'cover' ? title : s.title }));
