import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="flex items-center justify-between h-16 px-6 bg-white shadow-sm border-b border-gray-200">
      <div className="text-xl font-bold text-gray-900">tCATS</div>
      <div className="flex-1 max-w-2xl mx-6">
        <input
          type="text"
          placeholder="공연, 관객, 티켓 검색…"
          className="w-full px-4 py-2 text-sm border border-gray-200 rounded-lg bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </div>
      <div className="flex items-center space-x-4 text-gray-600">
        <button className="w-10 h-10 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200" aria-label="알림">
          <span className="text-lg">🔔</span>
        </button>
        <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center text-sm font-semibold text-gray-700">
          <span>JP</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
