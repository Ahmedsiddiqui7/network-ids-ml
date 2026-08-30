export function Card({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`border border-border bg-panel p-4 ${className}`}>
      {title && (
        <h3 className="mb-3 font-mono text-xs font-medium tracking-widest text-text-secondary uppercase">
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}
