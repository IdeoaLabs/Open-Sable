// Centralized icon exports to avoid Turbopack chunk loading issues
// This file pre-loads all icons to prevent dynamic import errors

export { 
  FiFile, 
  FiChevronRight, 
  FiChevronDown,
  FiGithub 
} from 'react-icons/fi';

export { 
  BsFolderFill, 
  BsFolder2Open 
} from 'react-icons/bs';

import { SiJavascript, SiReact, SiCss, SiJson } from 'react-icons/si';
const SiCss3 = SiCss;
export { SiJavascript, SiReact, SiCss3, SiJson };