import type { ReactNode } from 'react';

export function Card({
  children,
  className = '',
  testId,
  as: Tag = 'section',
  labelledBy,
}: {
  children: ReactNode;
  className?: string;
  testId?: string;
  as?: 'section' | 'div' | 'article';
  labelledBy?: string;
}) {
  return (
    <Tag className={`card ${className}`} data-testid={testId} aria-labelledby={labelledBy}>
      {children}
    </Tag>
  );
}

export function CardHeader({
  title,
  id,
  icon,
  subtitle,
  actions,
  className = '',
}: {
  title: ReactNode;
  id?: string;
  icon?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex flex-wrap items-start justify-between gap-x-4 gap-y-2 ${className}`}>
      <div className="flex min-w-0 items-start gap-2">
        {icon && <span className="mt-0.5 text-muted">{icon}</span>}
        <div className="min-w-0">
          <h2 id={id} className="text-base font-semibold">
            {title}
          </h2>
          {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold sm:text-[28px]">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-[15px] text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
