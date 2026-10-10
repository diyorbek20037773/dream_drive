import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { forwardRef } from 'react';

import { Spinner } from './Status';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'dark';
type Size = 'sm' | 'md';

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-white border-accent hover:bg-accent-strong hover:border-accent-strong',
  secondary: 'bg-surface text-ink border-line hover:bg-canvas',
  ghost: 'bg-transparent text-ink border-transparent hover:bg-canvas',
  danger: 'bg-danger text-white border-danger hover:brightness-110',
  dark: 'bg-white/10 text-white border-white/15 hover:bg-white/20',
};
const SIZES: Record<Size, string> = {
  sm: 'h-8 px-3 text-sm gap-1.5 rounded-[10px]',
  md: 'h-10 px-4 text-sm gap-2 rounded-control',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  busy?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon, busy = false, className = '', children, disabled, type, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type ?? 'button'}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={`inline-flex shrink-0 items-center justify-center border font-medium whitespace-nowrap transition-colors disabled:opacity-50 disabled:hover:brightness-100 ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...rest}
    >
      {busy ? <Spinner className="size-4" /> : icon}
      {children}
    </button>
  );
});
