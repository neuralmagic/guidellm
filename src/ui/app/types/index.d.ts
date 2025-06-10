declare module '*.ttf';

declare module '*.otf';

declare module '*.png' {
  const content: StaticImageData;
  export default content;
}
