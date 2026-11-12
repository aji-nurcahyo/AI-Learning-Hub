export type Route =
  | { type: 'home' | 'about' | 'missing' }
  | { type: 'getting-started'; module?: string }
  | { type: 'level' | 'chapter'; id: number; view?: string }
  | { type: 'track'; id: string; view?: 'practice' | 'capstone' };

export function parseRoute(hash: string): Route {
  const route = hash.replace(/^#\/?/, '');
  if (!route || route === 'curriculum' || route === 'tracks') return { type: 'home' };
  if (route === 'about') return { type: 'about' };
  if (route === 'getting-started' || route.startsWith('getting-started/')) {
    const mod = route.replace(/^getting-started\/?/, '');
    return { type: 'getting-started', module: mod || undefined };
  }
  const match = /^(level|chapter)\/(\d+)(?:\/(read|slides|template|task|practice))?$/.exec(route);
  if (match) {
    const id = Number(match[2]);
    if (id >= 1 && id <= (match[1] === 'level' ? 4 : 40) && !(match[1] === 'level' && match[3])) {
      return { type: match[1] as 'level' | 'chapter', id, view: match[3] };
    }
  }
  const track = /^track\/([a-z-]+)(?:\/(practice|capstone))?$/.exec(route);
  return track
    ? { type: 'track', id: track[1], view: track[2] as 'practice' | 'capstone' | undefined }
    : { type: 'missing' };
}
