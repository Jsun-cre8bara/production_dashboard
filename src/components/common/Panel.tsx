import React from 'react';

type PanelProps = {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
};

const Panel: React.FC<PanelProps> = ({ title, action, children }) => {
  return (
    <section className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        {action && <div className="flex items-center space-x-2">{action}</div>}
      </div>
      {children}
    </section>
  );
};

export default Panel;
